# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import os
import argparse
import re
import numpy as np
from collections import defaultdict
from transformers import AdamW, get_linear_schedule_with_warmup, T5ForConditionalGeneration
from utils import T5PegasusTokenizer, EncoderDecoderData, copy_loss, compute_rouge, compute_bleu, SRC_TOKEN, TGT_TOKEN
from t5_copy import T5Copy
from loguru import logger
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
new_tokens = ['，', '（', '）', '_', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def create_optimizer(model, lr, weight_decay, custom_lr=None):
    no_decay = 'bias|norm'
    params = defaultdict(list)
    custom_lr = custom_lr or dict()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        in_custom = False
        for custom_name, _ in custom_lr.items():
            if custom_name in name:
                if re.search(no_decay, name.lower()):
                    params[custom_name].append(param)
                else:
                    params[custom_name + '_decay'].append(param)
                in_custom = True
                break
        if not in_custom:
            if re.search(no_decay, name):
                params['normal'].append(param)
            else:
                params['normal_decay'].append(param)

    optimizer_grouped_parameters = []
    for k, v in params.items():
        param_lr = custom_lr.get(k.split('_')[0], lr)
        decay = weight_decay if 'decay' in k else 0.0
        optimizer_grouped_parameters.append({'params': v, 'weight_decay': decay, 'lr': param_lr}, )

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


class CopyT5Model():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = T5PegasusTokenizer.from_pretrained(args.model_path, do_lower_case=False)
        # add custom word
        self.tokenizer.add_tokens(new_tokens)
        self.model = T5Copy.from_pretrained(args.model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)

    def predict_batch(self, batch):
        pred = self.model.generate(
            eos_token_id=self.tokenizer.sep_token_id,
            decoder_start_token_id=self.tokenizer.cls_token_id,
            num_beams=3,
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
            max_length=self.args.max_target_length,
            src=batch['input_ids']
        )
        pred = pred[:, 1:].cpu().numpy()
        pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
        pred = [s.replace(' ', '') for s in pred]
        # logger.debug(f'inline pred: {pred}')
        return pred

    def validation_step(self, batch):
        ret = {}
        if self.args.compute_rouge:
            ret['rouge'] = 0
        if self.args.compute_bleu:
            ret['bleu'] = 0
        pred = self.predict_batch(batch)
        label = batch['decoder_input_ids'][:, 1:].cpu().numpy()
        label = self.tokenizer.batch_decode(label, skip_special_tokens=True)
        label = [s.replace(' ', '') for s in label]
        if self.args.compute_rouge:
            rouge = compute_rouge(label, pred, mode=args.rouge_mode)
            ret.update(rouge)
        if self.args.compute_bleu:
            bleu = compute_bleu(label, pred)
            ret['bleu'] = bleu
        return ret, pred

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.args.lr, self.args.weight_decay)
        if self.args.max_epochs == -1:
            t_total = self.args.max_steps // self.args.accumulate_grad_batches
        else:
            t_total = len(self.train_dataset) // self.args.accumulate_grad_batches * self.args.max_epochs
        if self.args.warmup_steps != -1:
            warmup_steps = self.args.warmup_steps
        else:
            warmup_steps = int(self.args.warmup_proportion * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        return optimizer, scheduler

    def save_model(self, output_dir, model, results=None):
        """
        Saves the model to output_dir.
        :param output_dir:
        :param model:
        :param results:
        :return:
        """
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved model checkpoint to {output_dir}")
        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def train_model(self, output_dir=None):
        """
        Trains the model using 'train_data'

        Args:
            train_data: Pandas DataFrame containing the 3 columns - `prefix`, `input_text`, `target_text`.
                        - `prefix`: A string indicating the task to perform. (E.g. `"question"`, `"stsb"`)
                        - `input_text`: The input text sequence. `prefix` is automatically prepended to form the full input. (<prefix>: <input_text>)
                        - `target_text`: The target sequence
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down training significantly as the predicted sequences need to be generated.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """
        data = EncoderDecoderData(self.args, self.tokenizer)
        dataloaders = data.get_dataloader()
        train_dataset, dev_dataset = dataloaders['train'][0], dataloaders['dev'][0]
        self.train_dataset = train_dataset
        t_steps = len(train_dataset)
        logger.debug(f'train_dataset: {type(train_dataset)}, {len(train_dataset)}')
        optimizer, scheduler = self.configure_optimizers()
        training_details = []
        min_loss = 1000
        for epoch in tqdm(range(0, self.args.max_epochs)):
            self.current_epoch = epoch
            self.model.train()
            batch_iterator = tqdm(
                train_dataset,
                desc=f"Running Epoch {epoch} of {self.args.max_epochs}",
                mininterval=0,
            )
            for batch_idx, batch in enumerate(batch_iterator):
                # logger.debug(f'batch: {batch}, {batch["input_ids"].shape}')
                logits = self.model(**batch).logits
                logger.info(f"batch labels: {batch['labels']}, {batch['decoder_attention_mask']}")
                loss = copy_loss(logits, batch['labels'], batch['decoder_attention_mask'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                training_details.append(loss.item())
                batch_iterator.set_description(
                    f"Epochs {epoch}/{self.args.max_epochs}. Training Loss: {loss.item():.4f}"
                )
            logger.debug(f"Epoch: {epoch}/{self.args.max_epochs}, train loss: {loss.item():.4f}")
            self.model.eval()
            val_iterator = tqdm(
                dev_dataset,
                desc=f"Running Epoch {epoch} of {self.args.max_epochs}",
                mininterval=0,
            )
            test_loss = 0
            val_evals = []
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_iterator):
                    logits = self.model(**batch).logits
                    loss = copy_loss(logits, batch['labels'], batch['decoder_attention_mask'])
                    test_loss += loss.item()
                    val_ret, val_pred = self.validation_step(batch)
                    val_evals.append(val_ret)
                    val_preds.extend(val_pred)
                    label = self.tokenizer.batch_decode(batch['labels'].cpu().numpy(), skip_special_tokens=True)
                    label = [s.replace(' ', '') for s in label]
                    val_labels.extend(label)
                    val_iterator.set_description(
                        f"Epochs {epoch}/{self.args.max_epochs}. Dev Loss: {loss.item():.4f}. Dev Eval: {val_ret}"
                    )
            test_loss /= len(list(dev_dataset))
            val_bleu = 0
            for i in val_evals:
                val_bleu += i.get('bleu', 0)
            val_bleu /= len(val_evals)
            num = 3
            logger.debug(f"y_pred: {val_preds[-num:]}")
            logger.debug(f"y_truth: {val_labels[-num:]}")
            logger.debug(f"Epoch: {epoch}/{self.args.max_epochs}, average dev loss: {test_loss:.4f}, average dev blue: {val_bleu:.4f}")
            training_details.append(test_loss)
            # Save model checkpoint
            output_dir = self.args.output_dir if output_dir is None else output_dir
            os.makedirs(self.args.output_dir, exist_ok=True)
            epoch_output_dir = os.path.join(output_dir, f'epoch_{epoch}')
            os.makedirs(epoch_output_dir, exist_ok=True)
            self.save_model(epoch_output_dir, self.model)
            if test_loss < min_loss:
                self.save_model(output_dir, self.model)
                min_loss = test_loss
            # Predict model
            # if data.predict_data:
            #     pred_batch = data.predict_data[-3:] 
            #     pred_source = [x[SRC_TOKEN] for x in pred_batch]
            #     pred_target = [x[TGT_TOKEN] for x in pred_batch]
            #     pred = predict(pred_source, output_dir)
            #     logger.debug('\n\npredict:')
            #     logger.debug(f'y_pred: {pred}')
            #     logger.debug(f'y_truth:{pred_target}')
        return training_details

def predict(texts, model_dir, batch_size=32, max_length=128, silent=True):
    tokenizer = T5PegasusTokenizer.from_pretrained(model_dir)
    model = T5Copy.from_pretrained(model_dir)
    model.to(device)
    logger.info('model loaded.')
    result = []
    for batch in tqdm([texts[i:i + batch_size] for i in range(0, len(texts), batch_size)],
                        desc="Generating outputs", disable=silent):
        inputs = tokenizer(batch, padding=True, max_length=max_length, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            # outputs = model.generate(inputs['input_ids'], max_length=max_length)
            pred = model.generate(
                eos_token_id=tokenizer.sep_token_id,
                decoder_start_token_id=tokenizer.cls_token_id,
                num_beams=3,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                src=inputs['input_ids']
            )
            pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
            pred = [s.replace(' ', '') for s in pred]
            result.extend(pred)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ========================= Train and trainer ==========================
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument('--plugins', type=str, default='ddp_sharded')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--kfold', type=int, default=1)
    parser.add_argument('--compute_bleu', action='store_true')
    parser.add_argument('--compute_rouge', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')

    # ========================= Data ==========================
    parser.add_argument('--train_file', type=str, required=False)
    parser.add_argument('--dev_file', type=str, required=False)
    parser.add_argument('--predict_file', type=str, required=False)
    parser.add_argument('--noise_prob', default=0., type=float)
    parser.add_argument('--max_source_length', default=200, type=int)
    parser.add_argument('--max_target_length', default=200, type=int)
    parser.add_argument('--beams', default=3, type=int)
    parser.add_argument('--num_works', type=int, default=4)

    # ========================= Model ==========================
    parser.add_argument('--model_path', type=str, default='imxly/t5-copy')
    parser.add_argument('--rouge_mode', type=str, default='all')
    parser.add_argument('--output_dir', type=str, default='saved/pytorch-checkpoints/')

    args = parser.parse_args()
    print(args)
    m = CopyT5Model(args)
    if args.do_train:
        m.train_model()
    if args.do_predict:
        sents = ['类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤']
        r = predict(sents, args.output_dir)
        print('input:', sents)
        print('output:', r)
