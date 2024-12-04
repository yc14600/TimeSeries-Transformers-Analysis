#export CUDA_VISIBLE_DEVICES=1

model_name=TimeTokenFormer
# List of random seeds to use (you can modify this)
seeds=(3)
for seed in "${seeds[@]}"
do
  echo "Running with seed: $seed"
  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Monthly' \
    --model_id m4_Monthly \
    --model $model_name \
    --data m4 \
    --features M \
    --e_layers 4 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 128 \
    --d_ff 128 \
    --te_layers 1 \
    --cp_heads 4 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 16 \
    --train_epochs 1 \
    --num_workers 1 \
    --cp_d_ff 64 \
    --n_heads 4 \
    --freq_energy_threshold 0.05 \
    --max_trend_num 8 \
    --max_freq_num 8 \
    --learning_rate 0.0001 \
    --loss 'SMAPE' \
    --seed $seed
done
