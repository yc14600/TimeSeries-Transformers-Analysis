import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_long_term_token_wrapper import Exp_Long_Term_Forecast_Token_Wrapper
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_short_term_token_wrapper import Exp_Short_Term_Forecast_Token_Wrapper
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.metrics import metric
from data_tokenizer import batch_tokenize, tokenize
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
from scipy.stats import spearmanr,levene,ttest_rel
import pandas as pd

def decompose_seasonal_residual(outputs_seasonal_residual,args,device):
    outputs_seasonal_tokens,outputs_seasonal,outputs_season_token_seqs = batch_tokenize.generate_seasonal_tokens(torch.tensor(outputs_seasonal_residual,dtype=torch.float32),energy_threshold=args.freq_energy_threshold,max_freq_num=args.max_freq_num,device=device)
    # outputs_seasonal = outputs_seasonal_components.cpu().numpy()
    outputs_residual = outputs - outputs_trend - outputs_seasonal
    outputs_seasonal_emb = outputs_seasonal_tokens.reshape(outputs_seasonal_tokens.shape[0],outputs_seasonal_tokens.shape[1],-1)
    return outputs_seasonal_emb,outputs_residual

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_a/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2021, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # Decomposition
    parser.add_argument('--filter_cutoff', type=float, default=0.02, help='filter cutoff for trend decomposition')
    parser.add_argument('--filter_order', type=int, default=5, help='filter order for trend decomposition')
    parser.add_argument('--double_smooth', type=bool, default=False, help='double smooth for trend decomposition')
    parser.add_argument('--fs', type=float, default=1.0, help='sampling frequency for seasonal decomposition')
    parser.add_argument('--trend_diff_thd', type=float, default=1e-3, help='threshold for trend difference')
    parser.add_argument('--freq_energy_threshold', type=float, default=0.05, help='threshold for frequency energy')
    parser.add_argument('--outlier_threshold', type=float, default=2.0, help='threshold for outlier detection')
    parser.add_argument('--max_trend_num', type=int, default=8, help='max number of trend segments')
    parser.add_argument('--max_freq_num', type=int, default=4, help='max number of frequency components')
       
    # TimeTokenFormer specific arguments
    parser.add_argument('--token_aggregate_func', type=str, default='mean', help='token aggregation function')  
    parser.add_argument('--te_layers', type=int, default=2, help='number of token encoder layers of TimeTokenFormer') 
    parser.add_argument('--cp_heads', type=int, default=4, help='number of heads of token component encoders')
    parser.add_argument('--cp_d_ff', type=int, default=128, help='feedfoward layer of token component encoders')
    parser.add_argument('--cp_d_model', type=int, default=96, help='embedding size of token component encoders')


    
    args = parser.parse_args()
    # fix_seed = 2021
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Seed:', seed)
    print('GPU available',torch.cuda.is_available())
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False
    if args.use_gpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)  # Optional to set memory fraction

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)
    
    print('Task:', args.task_name)

    if args.task_name == 'long_term_forecast':
        if args.model == 'TimeTokenFormer':
            Exp = Exp_Long_Term_Forecast_Token_Wrapper
        else:
            Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        print('Short Term Forecast')
        if args.model == 'TimeTokenFormer':
            Exp = Exp_Short_Term_Forecast_Token_Wrapper
        else:
            Exp = Exp_Short_Term_Forecast

    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast
    print('Exp:', Exp)
 
    ii = 0
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('device:', exp.device)    
    exp.model = exp.model.to(exp.device)
    exp.model.eval()
    test_data, test_loader = exp._get_data(flag='test')
    
    tot_rho = 0.
    tot_season_sim = 0.
    tot_v_p_value = 0.
    tot_t_p_value = 0.
    tot_mse = 0.
    tot_mae = 0.
    
    with torch.no_grad():
        for i, data_batch in enumerate(test_loader):
            outputs, batch_y = exp.model_feed_loop(data_batch)
            outputs = outputs.detach().permute(0,2,1)
            batch_y = batch_y.detach().permute(0,2,1)
            if test_data.scale and exp.args.inverse:
                shape = outputs.shape
                outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

            outputs_trend = tokenize.lowpass_filter(outputs,cutoff=args.filter_cutoff ,fs=args.fs,order=args.filter_order,device=exp.device)
            y_trend = tokenize.lowpass_filter(batch_y,cutoff=args.filter_cutoff,fs=args.fs,order=args.filter_order,device=exp.device)
            
            outputs_seasonal_residual = outputs-outputs_trend
            y_seasonal_residual = batch_y - y_trend
            
            outputs_seasonal_emb,outputs_residual = decompose_seasonal_residual(outputs_seasonal_residual,args,exp.device)
            y_seasonal_emb,y_residual = decompose_seasonal_residual(y_seasonal_residual,args,exp.device)
            
            outputs_trend = outputs_trend.cpu().numpy()
            y_trend = y_trend.cpu().numpy()
            outputs_residual = outputs_residual.cpu().numpy()
            y_residual = y_residual.cpu().numpy()
            
            ## evaluate the trend by spearman correlation
            rho=np.array([[spearmanr(a,b)[0] for (a,b) in zip(outputs_trend[b],y_trend[b])] for b in range(outputs_trend.shape[0])])
            rho = np.nanmean(rho,axis=0)

            ## evaluate the seasonal by cosine similarity
            season_sim = torch.einsum('bce,bce -> bc',outputs_seasonal_emb,y_seasonal_emb)
            norm = torch.sqrt((outputs_seasonal_emb**2).sum(dim=-1)) * torch.sqrt((y_seasonal_emb**2).sum(dim=-1))
            season_sim/=norm
            season_sim = np.nanmean(season_sim.cpu().numpy(),axis=0)
            
            ## evaluate the residual by mse, levene test and paired t-test
            v_p_value = np.array([[levene(a,b)[1] for (a,b) in zip(outputs_residual[b],y_residual[b])] for b in range(outputs_residual.shape[0])])
            t_p_value = np.array([[ttest_rel(a,b)[1] for (a,b) in zip(outputs_residual[b],y_residual[b])] for b in range(outputs_residual.shape[0])])
            
            v_p_value = np.nanmean(v_p_value,axis=0)
            t_p_value = np.nanmean(t_p_value,axis=0)
            
            mse = np.mean((outputs_residual-y_residual)**2,axis=(0,2))
            mae = np.mean(np.abs(outputs_residual-y_residual),axis=(0,2))

            tot_rho += rho
            tot_season_sim += season_sim
            tot_v_p_value += v_p_value            
            tot_t_p_value += t_p_value
            tot_mse += mse
            tot_mae += mae
            
        
        tot_rho /= (i+1)
        tot_season_sim /= (i+1)
        tot_v_p_value /= (i+1)
        tot_t_p_value /= (i+1)       
        tot_mse /= (i+1)       
        tot_mae /= (i+1)
        
    
    decompose_eval_features = pd.DataFrame({'rho':tot_rho,'season_sim':tot_season_sim,'v_p_value':tot_v_p_value,'t_p_value':tot_t_p_value,'mse':tot_mse,'mae':tot_mae})     
    decompose_eval_features.to_csv('./eval_results/decompose_eval_features_'+args.model_id+'_'+args.model+'.csv',index=False)       
            
            
            
    
    with open('./eval_results/decompose_results.txt','a') as f:
        f.write(setting + "  \n")
        f.write('rho:{},season sim:{},v_pvalue:{},t_pvalue:{},mse:{},mae:{}\n'.format(tot_rho.mean().round(3),tot_season_sim.mean().round(3),tot_v_p_value.mean().round(3),tot_t_p_value.mean().round(3),tot_mse.mean().round(3),tot_mae.mean().round(3)))
    
    torch.cuda.empty_cache()
