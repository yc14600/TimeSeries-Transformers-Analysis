model_name=Autoformer

SEEDS=(42 123 456)
PRED_LENS=(96 192 336 720)


for SEED in "${SEEDS[@]}"
do
  for PRED_LEN in "${PRED_LENS[@]}"
  do
    python -u est_mi.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/simulate/ \
      --data_path independent_alpha95.csv \
      --model_id simulate_idp_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --seed $SEED \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 2 \
      --dec_in 2 \
      --c_out 2 \
      --des 'Exp' \
      --itr 1 \
      --train_epochs 10 \


    python -u est_mi.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/simulate/ \
      --data_path dependent_alpha95_gamma20.csv \
      --model_id simulate_dp02_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --seed $SEED \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 2 \
      --dec_in 2 \
      --c_out 2 \
      --des 'Exp' \
      --itr 1 \
      --train_epochs 10


    python -u est_mi.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/simulate/ \
      --data_path dependent_alpha95_gamma40.csv \
      --model_id simulate_dp04_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --seed $SEED \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 2 \
      --dec_in 2 \
      --c_out 2 \
      --des 'Exp' \
      --itr 1 \
      --train_epochs 10


    python -u est_mi.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/simulate/ \
      --data_path dependent_alpha95_gamma60.csv \
      --model_id simulate_dp06_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --seed $SEED \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 2 \
      --dec_in 2 \
      --c_out 2 \
      --des 'Exp' \
      --itr 1 \
      --train_epochs 10


    python -u est_mi.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/simulate/ \
      --data_path dependent_alpha95_gamma80.csv \
      --model_id simulate_dp08_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --seed $SEED \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 2 \
      --dec_in 2 \
      --c_out 2 \
      --des 'Exp' \
      --itr 1 \
      --train_epochs 10

  done
done
