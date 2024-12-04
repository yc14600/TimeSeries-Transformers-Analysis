#export CUDA_VISIBLE_DEVICES=1

model_name=TimeTokenFormer
# List of random seeds to use (you can modify this)
seeds=(3)
for seed in "${seeds[@]}"
do
  echo "Running with seed: $seed"
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --d_model 128 \
    --d_ff 512 \
    --e_layers 4 \
    --d_layers 1 \
    --te_layers 1 \
    --cp_heads 4 \
    --factor 3 \
    --enc_in 862 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 8 \
    --train_epochs 5 \
    --num_workers 1 \
    --cp_d_ff 128 \
    --n_heads 4 \
    --freq_energy_threshold 0.05 \
    --max_trend_num 8 \
    --max_freq_num 8 \
    --seed $seed
done
