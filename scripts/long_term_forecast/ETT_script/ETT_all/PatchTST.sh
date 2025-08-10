model_name=PatchTST
SEEDS=(42 123 456)
PRED_LENS=(96 192 336 720)
GPU=1
for SEED in "${SEEDS[@]}"
do
  for PRED_LEN in "${PRED_LENS[@]}"
  do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --gpu $GPU \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --seed $SEED \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --n_heads 2 \
        --decoder_type noNorm \
        --itr 1
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --gpu $GPU \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --seed $SEED \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --n_heads 2 \
        --decoder_type noNorm \
        --itr 1

    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --gpu $GPU \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_96 \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --seed $SEED \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --n_heads 2 \
        --decoder_type noNorm \
        --itr 1

    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --gpu $GPU \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2_96 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --seed $SEED \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --n_heads 2 \
        --decoder_type noNorm \
        --itr 1

    python -u est_mi.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --gpu $GPU \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --seed $SEED \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --n_heads 2 \
        --decoder_type noNorm \
        --itr 1
    
    python -u est_mi.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --gpu $GPU \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --seed $SEED \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --n_heads 2 \
        --decoder_type noNorm \
        --itr 1

    python -u est_mi.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --gpu $GPU \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_96 \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --seed $SEED \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --n_heads 2 \
        --decoder_type noNorm \
        --itr 1

    python -u est_mi.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --gpu $GPU \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2_96 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --seed $SEED \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --n_heads 2 \
        --decoder_type noNorm \
        --itr 1
    done
done
