#export CUDA_VISIBLE_DEVICES=1

model_name=TimeTokenFormer
# List of random seeds to use (you can modify this)
seeds=(3)
for seed in "${seeds[@]}"
do
  echo "Running with seed: $seed"
  python -u run.py \
    --gpu 0 \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --d_model 512 \
    --d_ff 512 \
    --e_layers 4 \
    --d_layers 1 \
    --te_layers 1 \
    --cp_heads 4 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 8 \
    --train_epochs 10 \
    --num_workers 1 \
    --cp_d_ff 128 \
    --n_heads 16 \
    --freq_energy_threshold 0.05 \
    --max_trend_num 8 \
    --max_freq_num 8 \
    --learning_rate 0.001 \
    --seed $seed

  python -u run.py \
    --gpu 0 \
    --use_amp \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_192 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --d_model 512 \
    --d_ff 512 \
    --e_layers 4 \
    --d_layers 1 \
    --te_layers 1 \
    --cp_heads 4 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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


  python -u run.py \
    --gpu 0 \
    --use_amp \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_336 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --d_model 512 \
    --d_ff 512 \
    --e_layers 4 \
    --d_layers 1 \
    --te_layers 1 \
    --cp_heads 4 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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


  python -u run.py \
    --gpu 0 \
    --use_amp \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_720 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --d_model 512 \
    --d_ff 512 \
    --e_layers 4 \
    --d_layers 1 \
    --te_layers 1 \
    --cp_heads 4 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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
