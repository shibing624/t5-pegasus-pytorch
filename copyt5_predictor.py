# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import operator
import sys
import time
import os
from transformers import AutoTokenizer, T5ForConditionalGeneration, BertTokenizer
import jieba
from functools import partial
from tqdm import tqdm
import torch
from typing import List
from loguru import logger
from t5_copy import T5Copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


class CopyT5Predictor(object):
    def __init__(self, model_dir):
        bin_path = os.path.join(model_dir, 'pytorch_model.bin')
        if not os.path.exists(bin_path):
            model_dir = "shibing624/copyt5-base-chinese"
            logger.warning(f'local model {bin_path} not exists, use default HF model {model_dir}')
        t1 = time.time()
        if 'mengzi' in model_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            self.tokenizer = T5PegasusTokenizer.from_pretrained(model_dir)
        self.model = T5Copy.from_pretrained(model_dir)
        self.model.to(device)
        logger.debug("Use device: {}".format(device))
        logger.debug('Loaded model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))

    def batch_predict(self, texts: List[str], max_length: int = 128, batch_size: int = 64, silent: bool = False):
        """
        预测
        :param texts: list[str], sentence list
        :param max_length: int, max length of each sentence
        :param batch_size: int, bz
        :param silent: bool, show log
        :return: list, (corrected_text, [error_word, correct_word, begin_pos, end_pos])
        """
        result = []
        for batch in tqdm([texts[i:i + batch_size] for i in range(0, len(texts), batch_size)],
                          desc="Generating outputs", disable=silent):
            inputs = self.tokenizer(batch, padding=True, max_length=max_length, truncation=True,
                                    return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = self.model.generate(
                    eos_token_id=self.tokenizer.sep_token_id,
                    decoder_start_token_id=self.tokenizer.cls_token_id,
                    num_beams=3,
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    src=inputs['input_ids']
                )
            preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            preds = [s.replace(' ', '') for s in preds]
            result.extend(preds)
        return result


if __name__ == "__main__":
    m = CopyT5Predictor('saved/copyt5_ads')
    sentences = [
        "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤",
        "类型#裙*风格#简约*图案#条纹*图案#线条*图案#撞色*裙型#鱼尾裙*裙袖长#无袖",
        "类型#上衣*版型#宽松*颜色#粉红色*图案#字母*图案#文字*图案#线条*衣样式#卫衣*衣款式#不规则"
    ]
    logger.info(f"sents size: {len(sentences)}, \nhead: {sentences[:3]}")
    t2 = time.time()
    res = m.batch_predict(sentences)
    for sent, r in zip(sentences, res):
        print("{}=>{}".format(sent, r))
    logger.info(f'[batch]spend time: {time.time() - t2}')
