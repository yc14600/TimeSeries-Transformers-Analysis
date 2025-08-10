import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

def replace_each_feature(batch_x, replacements,mask,device):

    X_expanded = batch_x.unsqueeze(1).expand(-1, args.enc_in, -1, -1)
    replacements = replacements.to(device)
    X_expanded = X_expanded.to(device)

    X_replaced = torch.where(mask, replacements, X_expanded)
    return X_replaced

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
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--load2device', action='store_true', help='load whole dataset to device', default=False)

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
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--no_skip', default=False, action="store_true", help="NO skip connection in transformer")
    parser.add_argument('--fuse_decoder', default=False, action="store_true", help="Add a fuse layer to decoder projection")
    parser.add_argument('--decoder_type', type=str, default='conv2d', help="the type of the fuse layer in decoder projection, can be conv2d and MLP")
    
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
      
    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')


    
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Seed:', seed)
    print('GPU available',torch.cuda.is_available())
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
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        print('Short Term Forecast')
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
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_noSkip{}_FDC{}_{}_{}'.format(
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
        args.des, 
        args.no_skip,
        args.fuse_decoder,
        args.decoder_type,
        args.seed)
    mse, mae = 0., 0.
    exp = Exp(args)  # set experiments
    print('device:', exp.device)    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    print('loading model')
    exp.model.load_state_dict(torch.load(os.path.join(exp.args.checkpoints, setting,'checkpoint.pth'),map_location=exp.device))
    exp.model = exp.model.to(exp.device)
    exp.model.eval()
    print('model loaded')
    test_data, test_loader = exp._get_data(flag='test')
    self_mi, cross_mi = 0., 0.

    preds = []
    trues = []
    

    eye_mask = torch.eye(args.enc_in).to(exp.device)
    cross_mi_mt = 0.
    with torch.no_grad():
        for i, data_batch in enumerate(test_loader):
            
            batch_x,batch_y, batch_x_mark, dec_inp, batch_y_mark = exp.prepare_batch(data_batch)
            
            seq_indices = torch.arange(args.seq_len,device=exp.device).view(1, 1, args.seq_len, 1).expand(batch_x.shape[0], args.enc_in, -1, args.enc_in)  # Shape: [N, F, L, F]

            # Create a mask to identify positions where the feature index equals the last dimension index
            # This mask will be True where we need to apply the permutation
            feature_indices = torch.arange(args.enc_in,device=exp.device).view(1, args.enc_in, 1, 1).expand(batch_x.shape[0], -1, args.seq_len, args.enc_in)
            feature_indices_last = torch.arange(args.enc_in,device=exp.device).view(1, 1, 1, args.enc_in).expand(batch_x.shape[0], args.enc_in, args.seq_len, -1)
            mask = feature_indices == feature_indices_last  # Shape: [N, F, L, F]
            mask = mask.to(exp.device)
            
            expanded_shape = [batch_x.shape[0],args.enc_in,args.seq_len,args.enc_in]
            rp1 = torch.zeros(expanded_shape,device=exp.device)
            rp2 = torch.randn(expanded_shape,device=exp.device)

            rp3 = batch_x.unsqueeze(1).expand(-1, args.enc_in, -1, -1)
            
            rp4 = 0.5 * torch.randn(expanded_shape,device=exp.device) + 0.5 * rp3
            rp5 = 0.1 * torch.randn(expanded_shape,device=exp.device) + 0.9 * rp3
            
            batch_x_mark_expand = batch_x_mark.unsqueeze(1).expand(-1, args.enc_in, -1, -1)   
            batch_x_mark_expand = batch_x_mark_expand.reshape(batch_x.shape[0] * args.enc_in, args.seq_len, batch_x_mark.shape[-1])

            dec_inp = dec_inp.unsqueeze(1).expand(-1, args.enc_in, -1, -1) 
            dec_inp = dec_inp.reshape(batch_x.shape[0] * args.enc_in, dec_inp.shape[-2], dec_inp.shape[-1])

            batch_y_mark = batch_y_mark.unsqueeze(1).expand(-1, args.enc_in, -1, -1)  
            batch_y_mark = batch_y_mark.reshape(batch_x.shape[0] * args.enc_in, batch_y_mark.shape[-2], batch_y_mark.shape[-1])  
            
            new_outputs = torch.empty(5,batch_x.shape[0] * args.enc_in, args.pred_len, args.enc_in,device=exp.device)
            with torch.no_grad():
                for n,rp in enumerate([rp1,rp2, rp3,rp4,rp5]):
                    x_replaced = replace_each_feature(batch_x,rp,mask=mask,device=exp.device)

                    rp_inputs = x_replaced.view(batch_x.shape[0] * args.enc_in, args.seq_len, args.enc_in)
                    new_outputs[n] = exp.model(rp_inputs,batch_x_mark_expand,dec_inp,batch_y_mark)

            tot = torch.std(new_outputs,dim=0)
            tot = tot.reshape(batch_x.shape[0],args.enc_in,args.pred_len,args.enc_in)
            sdv = tot.mean(dim=0).mean(dim=1)
            self_mi += (sdv * eye_mask).sum()/sdv.shape[0]
            cross_mi += (sdv * (1-eye_mask)).sum()/(sdv.shape[0]*(sdv.shape[0]-1))
            cross_mi_mt += (sdv * (1-eye_mask))
    cross_mi_mt /= (i+1)
    max_cross_mi = (cross_mi_mt * (1-eye_mask)).max()        
    self_mi/=(i+1)
    cross_mi/=(i+1)
    print('iters',i+1,'Self MI:', self_mi, 'Cross MI:', cross_mi,'max cross MI', max_cross_mi)
    
    with open('./eval_results/mi_results.txt','a') as f:
        f.write(setting + "  \n")
        f.write('self_mi:{},cross_mi:{},mse:{},mae:{},max_cross_mi:{}\n'.format(self_mi,cross_mi,mse,mae,max_cross_mi))
    
    torch.cuda.empty_cache()
