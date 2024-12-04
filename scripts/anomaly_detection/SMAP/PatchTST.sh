model_name=PatchTST

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMAP/ \
  --model_id SMAP \
  --model $model_name \
  --data SMAP \
  --features 'M' \
  --seq_len 100 \
  --pred_len 0 \
  --anomaly_ratio 1 \
  --e_layers 3 \
  --batch_size 32 \
  --enc_in 25 \
  --c_out 25 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 10
