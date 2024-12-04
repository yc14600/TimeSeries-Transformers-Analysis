
model=PAttn

python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/simulate/ \
    --data_path independent.csv \
    --model_id 'simulate_idp_96_96' \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 96 \
    --train_epochs 10 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 1 \
    --d_layers 1 \
    --d_ff 512 \
    --enc_in 3 \
    --c_out 3 \
    --patch_size 16 \
    --stride 8 \
    --itr 1 \
    --des 'Exp' \
    --factor 3 \
    --model $model \
