

model_name=TimeXer
SEEDS=(42 123 456)
PRED_LENS=(96 192 336 720)
for SEED in "${SEEDS[@]}"
do
    for PRED_LEN in "${PRED_LENS[@]}"
    do
      python -u run.py \
        --task_name long_term_forecast \
        --gpu 1 \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id "ECL_96_$PRED_LEN" \
        --model $model_name \
        --seed $SEED \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --e_layers 3 \
        --factor 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --des 'Exp' \
        --d_ff 2048 \
        --batch_size 4 \
        --itr 1 
        
        python -u run.py \
        --task_name long_term_forecast \
        --gpu 1 \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id "ECL_96_$PRED_LEN" \
        --model $model_name \
        --seed $SEED \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --e_layers 3 \
        --factor 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --des 'Exp' \
        --d_ff 2048 \
        --batch_size 4 \
        --decoder_type noNorm \
        --itr 1
        
      python -u est_mi_slow.py \
        --task_name long_term_forecast \
        --gpu 1 \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id "ECL_96_$PRED_LEN" \
        --model $model_name \
        --seed $SEED \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --e_layers 3 \
        --factor 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --des 'Exp' \
        --d_ff 2048 \
        --batch_size 4 \
        --itr 1 
        
        python -u est_mi_slow.py \
        --task_name long_term_forecast \
        --gpu 1 \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id "ECL_96_$PRED_LEN" \
        --model $model_name \
        --seed $SEED \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --e_layers 3 \
        --factor 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --des 'Exp' \
        --d_ff 2048 \
        --batch_size 4 \
        --decoder_type noNorm \
        --itr 1        
        
    done
done