from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
# from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
# from utils.dtw_metric import dtw,accelerated_dtw
# from utils.augmentation import run_augmentation,run_augmentation_single
#from memory_profiler import memory_usage
from torch.nn import L1Loss,MSELoss

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.test_loader = None
        self.test_data = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag,drop_last=False):
        data_set, data_loader = data_provider(self.args, flag,drop_last=drop_last)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(vali_loader):
                outputs, batch_y = self.model_feed_loop(data_batch)
                loss = criterion(outputs, batch_y)
                loss = loss.detach().cpu().numpy()
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    
    
        
    def get_attentions(self,data_batch):
        batch_x,batch_y, batch_x_mark, dec_inp, batch_y_mark = self.prepare_batch(data_batch)
        attn = self.model.get_attention(batch_x, batch_x_mark)
        return attn
        
    
    
    
    def prepare_batch(self,data_batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        
        if 'PEMS' == self.args.data or 'Solar' == self.args.data:
            batch_x_mark = None
            batch_y_mark = None

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :],device=self.device).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # print('batch_x:',batch_x.shape,'batch_y:',batch_y.shape,'batch_x_mark:',batch_x_mark.shape,'dec_inp:',dec_inp.shape,'batch_y_mark:',batch_y_mark.shape)
        return batch_x,batch_y, batch_x_mark, dec_inp, batch_y_mark
    
    
    
    def model_feed_loop(self,data_batch,train=True):
        batch_x,batch_y, batch_x_mark, dec_inp, batch_y_mark = self.prepare_batch(data_batch)

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        return outputs,batch_y
    

    def train_loop(self,i,epoch,train_steps,data_batch,time_now,model_optim,criterion,scaler,train_loss,**kwargs):
        
        model_optim.zero_grad()
        
        outputs, batch_y = self.model_feed_loop(data_batch,train=True)
        # print('outputs:',outputs.shape,'batch_y:',batch_y.shape)
        loss = criterion(outputs, batch_y)
        train_loss.append(loss.item())

        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed = (time.time() - time_now) / (i+1)
            left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

        if self.args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            loss.backward()
            model_optim.step()
            
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self.test_loader = test_loader
        self.test_data = test_data
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, data_batch in enumerate(train_loader):
                # print('batch',i)
                self.train_loop(i,epoch,train_steps,data_batch,epoch_time,model_optim,criterion,scaler,train_loss)               
                
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
        best_model_path = path + '/' + 'checkpoint.pth'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path,map_location=self.device))

        return self.model

    def test(self, setting, test=0,root_path='./'):
        if self.test_data is None:
            test_data, test_loader = self._get_data(flag='test')               
        else:
            test_data, test_loader = self.test_data, self.test_loader
        # test_data.data_x = test_data.data_x.to(self.device)
        # test_data.data_y = test_data.data_y.to(self.device)
        # test_data.data_stamp = test_data.data_stamp.to(self.device)
        if test:
            model_path = os.path.join(self.args.checkpoints, setting,'checkpoint.pth')
            if os.path.exists(model_path):
                print('loading model')
                self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting,'checkpoint.pth'),map_location=self.device))
                self.model = self.model.to(self.device)
        preds = []
        trues = []
        folder_path = os.path.join(root_path,'test_results/',setting + '/')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        tot_mae,tot_mse = 0.,0.
        self.model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(test_loader):
                outputs, batch_y = self.model_feed_loop(data_batch)
                outputs = outputs.detach()
                batch_y = batch_y.detach()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                pred = outputs
                true = batch_y
                mae_i = L1Loss()(pred, true)
                mse_i = MSELoss()(pred, true)
                tot_mae += mae_i
                tot_mse += mse_i
        mae = tot_mae / len(test_loader)
        mse = tot_mse / len(test_loader)
        # preds = torch.concat(preds, dim=0)
        # trues = torch.concat(trues, dim=0)
        # print('test shape:', preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # if self.args.data == 'PEMS':
        #     B, T, C = preds.shape
        #     preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
        #     trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)
        # result save
        folder_path = os.path.join(root_path,'results',setting + '/') 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        # if self.args.use_dtw:
        #     dtw_list = []
        #     manhattan_distance = lambda x, y: np.abs(x - y)
        #     for i in range(preds.shape[0]):
        #         x = preds[i].reshape(-1,1)
        #         y = trues[i].reshape(-1,1)
        #         if i % 100 == 0:
        #             print("calculating dtw iter:", i)
        #         d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
        #         dtw_list.append(d)
        #     dtw = np.array(dtw_list).mean()
        # else:
        #     dtw = -999
            

        # mae, mse, rmse, mape, mspe = metric(preds.cpu().numpy(), trues.cpu().numpy())
        print('mse:{}, mae:{}'.format(mse, mae))
        # print('mse{},mae:{}, mape1:{}, rmse:{},mspe:{}'.format(mse,mae, mape, rmse,mspe))            
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        # if self.args.data == 'PEMS':
        #     f.write('mse{},mae:{}, mape:{}, rmse:{}'.format(mse,mae, mape, rmse))
        # else:
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return mse, mae
    

