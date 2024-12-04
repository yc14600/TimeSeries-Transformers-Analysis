model_name=PatchTST

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT/ \
  --model_id SWAT \
  --model $model_name \
  --data SWAT \
  --features 'M' \
  --seq_len 100 \
  --pred_len 0 \
  --anomaly_ratio 1 \
  --e_layers 3 \
  --batch_size 32 \
  --enc_in 51 \
  --c_out 51 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10
