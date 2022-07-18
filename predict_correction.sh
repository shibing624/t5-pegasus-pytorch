python predict_t5_copy.py \
--predict_file correction_data/test.json \
--batch_size 16 \
--max_source_length 500 \
--max_target_length 200 \
--model_path imxly/t5-copy \
--gpus 1 \
--output_path c_predictions.txt \
--resume saved/t5_copy-noise=0.0-0-epoch=04-bleu=0.8856-rouge-1=0.9722-rouge-2=0.9434-rouge-l=0.9718.ckpt