python train_t5_copy.py --train_file correction_data/train.json --dev_file correction_data/test.json \
--batch_size 16 \
--max_epochs 10 \
--max_source_length 500 \
--max_target_length 200 \
--model_path imxly/t5-copy \
--compute_bleu --compute_rouge \
--gpus 1 \
--lr 5e-5
