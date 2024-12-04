import torch
from .exp_short_term_forecasting import Exp_Short_Term_Forecast
from data_tokenizer.batch_tokenize import TimeTokenizer
from data_tokenizer.tokenize import lowpass_filter
import numpy as np

class Exp_Short_Term_Forecast_Token_Wrapper(Exp_Short_Term_Forecast):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast_Token_Wrapper, self).__init__(args)
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
        
        batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
        
        batch_x = batch_x.float().to(self.device)

        batch_y = batch_y.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # outputs = self.model(batch_x, None, dec_inp, None)

        batch_x = batch_x.permute(0,2,1)
        # batch_y = batch_y.permute(0,2,1)
        batch_trend_x = self.__decompose__(batch_x)
        batch_x = batch_x.float().to(self.device)
        batch_trend_x = batch_trend_x.float().to(self.device) 
                     
        # print('batch_x:',batch_x.shape,'batch_trend_x:',batch_trend_x.shape,'batch_y:',batch_y.shape,'dec_inp:',dec_inp.shape)
        trend_x_token,trend_x_time, seasonal_x_token,seasonal_x_time = self.tokenizer.generate_token_embeddings(batch_x,batch_trend_x,device=self.device)
        # print('check token nan',torch.isnan(trend_x_token).sum(),torch.isnan(trend_x_time).sum(),torch.isnan(seasonal_x_token).sum(),torch.isnan(seasonal_x_time).sum())
    
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(trend_x_token,trend_x_time, seasonal_x_token,seasonal_x_time,batch_x)[0]
                

        else:
            outputs = self.model(trend_x_token,trend_x_time, seasonal_x_token,seasonal_x_time, batch_x)[0]
        # print('outputs nan:',torch.isnan(outputs).sum())    
        outputs = outputs.permute(0,2,1)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)        
        batch_x = batch_x.permute(0,2,1)
        
        return outputs,batch_x, batch_y, batch_y_mark
    
    
    def model_feed_all(self, x,batch_size=500):
        B, _, C = x.shape
        dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
        dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
        # encoder - decoder
        outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
        id_list = np.arange(0, B, batch_size)  # validation set size
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            bx = x[id_list[i]:id_list[i + 1]]
            bdecinp = dec_inp[id_list[i]:id_list[i + 1]]
            bx = bx.permute(0,2,1)

            batch_trend_x = self.__decompose__(bx)
            bx = bx.float().to(self.device)
            batch_trend_x = batch_trend_x.float().to(self.device) 
            
            trend_x_token,trend_x_time, seasonal_x_token,seasonal_x_time = self.tokenizer.generate_token_embeddings(bx,batch_trend_x,device=self.device)

            # print('check token nan',torch.isnan(trend_x_token).sum(),torch.isnan(trend_x_time).sum(),torch.isnan(seasonal_x_token).sum(),torch.isnan(seasonal_x_time).sum())
            outputs[id_list[i]:id_list[i + 1], :, :] = self.model(trend_x_token,trend_x_time, seasonal_x_token,seasonal_x_time,bx)[0].detach().cpu().permute(0,2,1)
        # print('outputs nan:',torch.isnan(outputs).sum())
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:] 
        
        return outputs