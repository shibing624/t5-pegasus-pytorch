#predict_t5_copypython predict_t5_copy.py \
#--predict_file qg.jsonl \
#--batch_size 6 \
#--max_source_length 512 \
#--max_target_length 200 \
#--model_path /home/xianglingyang/pretrained_models/torch/t5-copy \
#--gpus 4 \
#--resume saved/0-epoch=05-val_bleu=0.0047.ckpt \
#--output_path qg_predictions.txt


python predict_t5_copy.py \
--predict_file data/sample_data.json \
--batch_size 16 \
--max_source_length 500 \
--max_target_length 200 \
--model_path imxly/t5-copy \
--gpus 1 \
--output_path predictions.txt \
--resume saved/t5_copy-noise=0.0-0-epoch=04-bleu=0.8856-rouge-1=0.9722-rouge-2=0.9434-rouge-l=0.9718.ckpt
