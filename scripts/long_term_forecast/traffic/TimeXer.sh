model_name=TimeXer
SEEDS=(42 123 456)
PRED_LENS=(96 192 336 720)

for SEED in "${SEEDS[@]}"
do
  for PRED_LEN in "${PRED_LENS[@]}"
  do
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id traffic_96 \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --seed $SEED \
        --e_layers 2 \
        --factor 3 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --d_model 512 \
        --d_ff 512 \
        --des 'Exp' \
        --batch_size 16 \
        --learning_rate 0.001 \
        --itr 1
  done
done

