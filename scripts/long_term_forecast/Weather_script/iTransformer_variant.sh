model_name=iTransformer

SEEDS=(42 123 456 789 1011)
PRED_LENS=(96 192 336 720)



for SEED in "${SEEDS[@]}"
do
  for PRED_LEN in "${PRED_LENS[@]}"
  do
    python -u est_mi.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --seed $SEED \
      --e_layers 3 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --d_model 512\
      --d_ff 512\
      --itr 1 \
      --train_epochs 10

    python -u est_mi.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --seed $SEED \
      --e_layers 3 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --d_model 512\
      --d_ff 512\
      --itr 1 \
      --train_epochs 10 \
      --fuse_decoder \
      --no_skip

    python -u est_mi.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --seed $SEED \
      --e_layers 3 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --d_model 512\
      --d_ff 512\
      --itr 1 \
      --train_epochs 10 \
      --no_skip 

    python -u est_mi.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --seed $SEED \
      --e_layers 3 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --d_model 512\
      --d_ff 512\
      --itr 1 \
      --train_epochs 10 \
      --fuse_decoder 
  done
done
