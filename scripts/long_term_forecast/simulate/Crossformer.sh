model_name=Crossformer

python -u est_mi.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/simulate/ \
  --data_path independent.csv \
  --model_id simulate_idp_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1

python -u est_mi.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/simulate/ \
  --data_path dependent02.csv \
  --model_id simulate_dp02_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1


python -u est_mi.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/simulate/ \
  --data_path dependent04.csv \
  --model_id simulate_dp04_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1


python -u est_mi.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/simulate/ \
  --data_path dependent06.csv \
  --model_id simulate_dp06_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1


python -u est_mi.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/simulate/ \
  --data_path dependent08.csv \
  --model_id simulate_dp08_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1


python -u est_mi.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/simulate/ \
  --data_path dependent10.csv \
  --model_id simulate_dp10_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1


