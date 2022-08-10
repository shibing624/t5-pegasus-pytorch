#python train_t5_copy.py \
#--train_file sample_data.json \  # 训练数据
#--dev_file sample_data.json \ # 验证数据
#--batch_size 6 \
#--max_epochs 15 \
#--max_source_length 512 \
#--max_target_length 200 \
#--model_path /home/xianglingyang/pretrained_models/torch/t5-copy \
#--compute_bleu \   # 是够计算bleu
#--compute_rouge \  # 是否计算rouge
#--gpus 4 \
#--lr 5e-5


python train_t5_copy_torch.py --train_file data/dev.json \
--predict_file data/sample_data.json \
--batch_size 32 \
--max_epochs 50 \
--max_source_length 200 \
--max_target_length 200 \
--model_path outputs/pytorch-checkpoints-ecommerv1/ \
--compute_bleu --compute_rouge \
--gpus 1 \
--lr 1e-4 \
--num_works 1 \
--output_dir imxly/t5-copy \
--do_train --do_predict
