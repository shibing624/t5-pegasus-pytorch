import os
import argparse
import pytorch_lightning as pl
import re
import numpy as np
from loguru import logger
from transformers import AdamW, T5Tokenizer
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup
from utils import T5PegasusTokenizer, EncoderDecoderData, copy_loss, compute_rouge, compute_bleu
from t5_copy import T5Copy
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


class TaskLightModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = T5Copy.from_pretrained(args.model_path)
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        logits = self(**batch).logits
        loss = copy_loss(logits, batch['labels'], batch['decoder_attention_mask'])
        return loss

    def predict_batch(self, batch):
        pred = self.model.generate(
            eos_token_id=tokenizer.sep_token_id,
            decoder_start_token_id=tokenizer.cls_token_id,
            num_beams=3,
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
            use_cache=True,
            max_length=self.args.max_target_length,
            src=batch['input_ids']
        )
        pred = pred[:, 1:].cpu().numpy()
        pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
        pred = [s.replace(' ', '') for s in pred]
        return pred

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.predict_batch(batch)

    def validation_step(self, batch, batch_idx):
        ret = {}
        if self.args.compute_rouge:
            ret['rouge'] = 0
        if self.args.compute_bleu:
            ret['bleu'] = 0
        if self.current_epoch + 1 < self.args.eval_start:
            return ret
        pred = self.predict_batch(batch)
        label = batch['decoder_input_ids'][:, 1:].cpu().numpy()
        label = tokenizer.batch_decode(label, skip_special_tokens=True)
        label = [s.replace(' ', '') for s in label]
        if self.args.compute_rouge:
            rouge = compute_rouge(label, pred, mode=args.rouge_mode)
            ret.update(rouge)
        if self.args.compute_bleu:
            bleu = compute_bleu(label, pred)
            ret['bleu'] = bleu
        return ret

    def validation_epoch_end(self, outputs):
        ret = {}
        if self.args.compute_rouge:
            ret['rouge'] = 0
        if self.args.compute_bleu:
            ret['bleu'] = 0
        if self.current_epoch + 1 < self.args.eval_start:
            return ret
        keys = outputs[0].keys()
        ret = {k: np.mean([x[k] for x in outputs]) for k in keys}
        for k, v in ret.items():
            self.log(k, v, prog_bar=True)
        print(ret)
        return ret

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.args.lr, self.args.weight_decay)
        if self.args.max_epochs == -1:
            t_total = self.args.max_steps // self.args.accumulate_grad_batches
        else:
            t_total = len(self.train_dataloader()) // self.args.accumulate_grad_batches * self.args.max_epochs
        if self.args.warmup_steps != -1:
            warmup_steps = self.args.warmup_steps
        else:
            warmup_steps = int(self.args.warmup_proportion * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ========================= Train and trainer ==========================
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--eval_start', default=3, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--plugins', type=str, default='ddp_sharded')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--kfold', type=int, default=1)
    parser.add_argument('--compute_bleu', action='store_true')
    parser.add_argument('--compute_rouge', action='store_true')

    # ========================= Data ==========================
    parser.add_argument('--train_file', type=str, required=False)
    parser.add_argument('--dev_file', type=str, required=False)
    parser.add_argument('--predict_file', type=str, required=False)
    parser.add_argument('--noise_prob', default=0., type=float)
    parser.add_argument('--max_source_length', default=200, type=int)
    parser.add_argument('--max_target_length', default=150, type=int)
    parser.add_argument('--beams', default=3, type=int)
    parser.add_argument('--num_works', type=int, default=4)

    # ========================= Model ==========================
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--rouge_mode', type=str, default='all')
    parser.add_argument('--save_path', type=str, default='./saved')
    parser.add_argument('--save_to_hf', action='store_true')

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if 'mengzi' in args.model_path:
        tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    else:
        tokenizer = T5PegasusTokenizer.from_pretrained(args.model_path)
    # add custom word
    tokenizer.add_tokens(['，', '（', '）'])
    
    data = EncoderDecoderData(args, tokenizer)
    dataloaders = data.get_dataloader()
    pl.seed_everything(args.seed)
    train_data, dev_data = dataloaders['train'][0], dataloaders['dev'][0]
    model = TaskLightModel(args)
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=args.save_path,
        filename='t5_copy-noise={}-{}-'.format(
            args.noise_prob, 0) + "{epoch:02d}-{bleu:.4f}-{rouge-1:.4f}-{rouge-2:.4f}-{rouge-l:.4f}",
        save_weights_only=True,
        save_on_train_epoch_end=True,
        monitor='bleu',
        mode='max',
    )
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint], logger=False)
    trainer.fit(model, train_data, dev_data)
    
    logger.info('train model done.')
    
    if args.save_to_hf:
        # 模型转为transformers可加载
        ckpt_path = os.path.join(args.save_path, 'copyt5_correction')
        os.makedirs(ckpt_path, exist_ok=True)
        logger.info(f'ckpt_path: {ckpt_path}')
        # model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        # 先保存原始transformer bert model
        tokenizer.save_pretrained(ckpt_path)
        model.model.save_pretrained(ckpt_path)
        # state_dict = torch.load(ckpt_path)['state_dict']
        # 再保存finetune训练后的模型文件，替换原始的pytorch_model.bin
        # torch.save(state_dict, os.path.join(args.save_path, 'pytorch_model.bin'))
