model_name=iTransformer

SEEDS=(42 123 456 789 1011)
PRED_LENS=(96 192 336 720)



for SEED in "${SEEDS[@]}"
do
  for PRED_LEN in "${PRED_LENS[@]}"
  do
    # python -u est_mi.py \
    #   --task_name long_term_forecast \
    #   --is_training 1 \
    #   --root_path ./dataset/ETT-small/ \
    #   --data_path ETTm2.csv \
    #   --model_id ETTm2_96_96 \
    #   --model $model_name \
    #   --data ETTm2 \
    #   --features M \
    #   --seq_len 96 \
    #   --label_len 48 \
    #   --pred_len $PRED_LEN \
    #   --seed $SEED \
    #   --e_layers 2 \
    #   --d_layers 1 \
    #   --factor 3 \
    #   --enc_in 7 \
    #   --dec_in 7 \
    #   --c_out 7 \
    #   --des 'Exp' \
    #   --d_model 128\
    #   --d_ff 128\
    #   --itr 1 \
    #   --train_epochs 10

    python -u est_mi.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm2.csv \
      --model_id ETTm2_96_96 \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --seed $SEED \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 128\
      --d_ff 128\
      --itr 1 \
      --train_epochs 10 \
      --fuse_decoder \
      --decoder_type 'MLP' \
      --no_skip

    # python -u est_mi.py \
    #   --task_name long_term_forecast \
    #   --is_training 1 \
    #   --root_path ./dataset/ETT-small/ \
    #   --data_path ETTm2.csv \
    #   --model_id ETTm2_96_96 \
    #   --model $model_name \
    #   --data ETTm2 \
    #   --features M \
    #   --seq_len 96 \
    #   --label_len 48 \
    #   --pred_len $PRED_LEN \
    #   --seed $SEED \
    #   --e_layers 2 \
    #   --d_layers 1 \
    #   --factor 3 \
    #   --enc_in 7 \
    #   --dec_in 7 \
    #   --c_out 7 \
    #   --des 'Exp' \
    #   --d_model 128\
    #   --d_ff 128\
    #   --itr 1 \
    #   --train_epochs 10 \
    #   --no_skip 

    python -u est_mi.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm2.csv \
      --model_id ETTm2_96_96 \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --seed $SEED \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 128\
      --d_ff 128\
      --itr 1 \
      --train_epochs 10 \
      --fuse_decoder \
      --decoder_type 'MLP' 
  done
done
