model_name=TimeTokenFormer
# List of random seeds to use (you can modify this)
seeds=(3)
for seed in "${seeds[@]}"
do
  echo "Running with seed: $seed"
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --d_model 128 \
    --d_ff 512 \
    --e_layers 4 \
    --d_layers 1 \
    --te_layers 1 \
    --cp_heads 4 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 96 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 8 \
    --train_epochs 10 \
    --num_workers 1 \
    --cp_d_ff 128 \
    --n_heads 4 \
    --learning_rate 0.0001 \
    --max_trend_num 8 \
    --max_freq_num 8 \
    --seed $seed
done

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --d_model 1024 \
#   --e_layers 4 \
#   --d_layers 1 \
#   --te_layers 2 \
#   --cp_heads 4 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 96 \
#   --c_out 21 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 3 \
#   --num_workers 1 \
#   --d_ff 512 \
#   --n_heads 16 \
#   --seed 0


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --itr 1
