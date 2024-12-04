import torch
from .exp_long_term_forecasting import Exp_Long_Term_Forecast
from data_tokenizer.batch_tokenize import TimeTokenizer
from data_tokenizer.tokenize import lowpass_filter

class Exp_Long_Term_Forecast_Token_Wrapper(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Token_Wrapper, self).__init__(args)
        self.decompose_args = {'filter_cutoff':args.filter_cutoff,'filter_order':args.filter_order,\
                          'double_smooth':args.double_smooth,'fs':args.fs,'trend_diff_thd':args.trend_diff_thd,\
                          'freq_energy_threshold':args.freq_energy_threshold,'outlier_threshold':args.outlier_threshold,\
                            'max_trend_num':args.max_trend_num,'max_freq_num':args.max_freq_num}
        self.tokenizer = TimeTokenizer(self.args.seq_len,self.args.enc_in,feature_names=None,decompose_args=self.decompose_args)
        
    def __decompose__(self,x):
        trend_x = lowpass_filter(x,self.decompose_args['filter_cutoff'],self.decompose_args['fs'],self.decompose_args['filter_order'])
        # print('trend_x:',trend_x.shape)
        return torch.from_numpy(trend_x.copy())

    def model_feed_loop(self,data_batch):
        
        batch_x, batch_y, stamp = data_batch
        batch_y = batch_y.float().to(self.device)
        stamp = stamp.float().to(self.device) if self.args.data != 'PEMS' else None

        batch_x = batch_x.permute(0,2,1)
        batch_y = batch_y.permute(0,2,1)
        batch_trend_x = self.__decompose__(batch_x)
        batch_x = batch_x.float().to(self.device)
        batch_trend_x = batch_trend_x.float().to(self.device) 
                     
        # print('batch_x:',batch_x.shape,'batch_trend_x:',batch_trend_x.shape,'batch_y:',batch_y.shape,'stamp:',stamp.shape)
        trend_x_token,trend_x_time, seasonal_x_token,seasonal_x_time = self.tokenizer.generate_token_embeddings(batch_x,batch_trend_x,device=self.device)
        # print('check nan',torch.isnan(trend_x_token).sum(),torch.isnan(trend_x_time).sum(),torch.isnan(seasonal_x_token).sum(),torch.isnan(seasonal_x_time).sum())
        # print('check inf',torch.isinf(trend_x_token).sum(),torch.isinf(trend_x_time).sum(),torch.isinf(seasonal_x_token).sum(),torch.isinf(seasonal_x_time).sum())
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                dec_out = self.model(trend_x_token,trend_x_time, seasonal_x_token,seasonal_x_time,batch_x, stamp)[0]
                

        else:
            dec_out = self.model(trend_x_token,trend_x_time, seasonal_x_token,seasonal_x_time, batch_x,stamp)[0]
            

        dec_out = dec_out.permute(0,2,1)
        batch_y = batch_y.permute(0,2,1)
        
        f_dim = -1 if self.args.features == 'MS' else 0
        dec_out = dec_out[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        
        return dec_out,batch_y