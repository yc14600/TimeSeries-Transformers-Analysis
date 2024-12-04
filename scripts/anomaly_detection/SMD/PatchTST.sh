model_name=PatchTST

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD/ \
  --model_id SMD \
  --model $model_name \
  --data SMD \
  --features 'M' \
  --seq_len 100 \
  --pred_len 0 \
  --anomaly_ratio 0.5 \
  --e_layers 3 \
  --batch_size 32 \
  --enc_in 38 \
  --c_out 38 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 10
