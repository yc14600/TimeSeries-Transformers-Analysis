import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import TransformerEncoderLayer, _get_activation_fn
from torch.nn import MultiheadAttention 
from torch.nn import Dropout,Linear,LayerNorm
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
import math
import os
import sys

here = os.getcwd()
sys.path.append(os.path.join(here,"../"))

from typing import Optional, Any, Union, Callable

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import TemporalEmbedding,TimeFeatureEmbedding
from data_tokenizer.tokenize import *
import gc


trend_token_dim = 8
season_token_dim = 4
residual_token_dim = 8

# out = self.out_proj(torch.cat([self.recompose(x_token),self.recompose(x_time)],dim=-1))



class TimeTokenCrossEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model: int, d_time: int, n_heads: int = 1,dim_feedforward: int = 512, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,  
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,device=None, dtype=None)->None:
        nn.Module.__init__(self)
        factory_kwargs = {'device': device, 'dtype': dtype}
        #nn.Sequential(self.norm0,Linear(emb_in, d_model, **factory_kwargs))
        self.trend_attn_layer = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
        self.season_attn_layer = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
        self.cross_attn_layer1 = MultiheadAttention(d_time, n_heads, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
        self.cross_attn_layer2 = MultiheadAttention(d_time, n_heads, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
        self.v_dim = int(d_time/n_heads)
        
        self.d_model = d_model
        self.d_time = d_time
        self.n_heads = n_heads

        self.dropout = Dropout(dropout)
        self.linear1 = Linear(d_model+d_time, dim_feedforward, **factory_kwargs)
        self.linear2 = Linear(dim_feedforward, d_model+d_time, **factory_kwargs)
        
        self.linear3 = Linear(d_model+d_time, dim_feedforward, **factory_kwargs)
        self.linear4 = Linear(dim_feedforward, d_model+d_time, **factory_kwargs)


        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_time, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm4 = LayerNorm(d_time, eps=layer_norm_eps, **factory_kwargs)
        self.norm5 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm6 = LayerNorm(d_time, eps=layer_norm_eps, **factory_kwargs)
        self.norm7 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm8 = LayerNorm(d_time, eps=layer_norm_eps, **factory_kwargs)
        
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

        
     
    ### to do: if no change in the forward function, we can use the parent class's forward function    
    def forward(self, trend_token: Tensor, trend_time: Tensor, season_token: Tensor, season_time: Tensor, 
                trend_src_mask: Optional[Tensor] = None,season_src_mask: Optional[Tensor] = None,
                trend_src_key_padding_mask: Optional[Tensor] = None,season_src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor,Tensor]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if trend_src_key_padding_mask is not None:
            _skpm_dtype = trend_src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(trend_src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of trend key_padding_mask are supported")
        if season_src_key_padding_mask is not None:
            _skpm_dtype = season_src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(season_src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of season key_padding_mask are supported")
               
        if self.norm_first:
            trend_token_o, trend_time_o, season_token_o, season_time_o = self._sa_block(self.norm1(trend_token),self.norm2(trend_time),self.norm3(season_token),self.norm4(season_time), 
                                                                                        trend_src_mask, trend_src_key_padding_mask,season_src_mask, season_src_key_padding_mask) 
            
            trend_token = trend_token+trend_token_o 
            trend_time = trend_time+trend_time_o 
            season_token = season_token + season_token_o
            season_time = season_time+season_time_o  
            
            trend_token_o, trend_time_o, season_token_o, season_time_o = self._ff_block(self.norm5(trend_token),self.norm6(trend_time),self.norm7(season_token),self.norm8(season_time))
                 
            trend_token = trend_token+trend_token_o 
            trend_time = trend_time+trend_time_o 
            season_token = season_token+season_token_o
            season_time = season_time+season_time_o
            
        else:
            # print('x',x.shape)
            # print('sax',self._sa_block(x, src_mask, src_key_padding_mask).shape)
            trend_token_o, trend_time_o, season_token_o, season_time_o = self._sa_block(trend_token,trend_time,season_token,season_time,trend_src_mask,trend_src_key_padding_mask,season_src_mask,season_src_key_padding_mask)
            trend_token = self.norm1(trend_token_o + trend_token)
            trend_time = self.norm2(trend_time_o + trend_time)
            season_token = self.norm3(season_token_o + season_token)
            season_time = self.norm4(season_time_o + season_time)
            
            trend_token_o, trend_time_o, season_token_o, season_time_o = self._ff_block(trend_token,trend_time,season_token,season_time)
            
            trend_token = self.norm5(trend_token_o + trend_token)
            trend_time = self.norm6(trend_time_o + trend_time)
            season_token = self.norm7(season_token_o + season_token)
            season_time = self.norm8(season_time_o + season_time)

        return trend_token,trend_time,season_token,season_time

    # self-attention block
    def _sa_block(self, trend_token: Tensor, trend_time:Tensor,season_token: Tensor, season_time:Tensor,
                  trend_attn_mask: Optional[Tensor], trend_key_padding_mask: Optional[Tensor],
                  season_attn_mask: Optional[Tensor], season_key_padding_mask: Optional[Tensor]) -> Tuple[Tensor,Tensor]:
        
        trend_token_attn_output = self.trend_attn_layer(trend_token, trend_token, trend_token,
                                            attn_mask=trend_attn_mask,
                                            key_padding_mask=trend_key_padding_mask,
                                            need_weights=True,average_attn_weights=False)
        
        # print('v_dim',v_dim,token_attn_output[1][:,0,...].shape,x_time.shape)
        trend_time_token_attn_output = (torch.einsum('bll,blt->blt',trend_token_attn_output[1][:,i,...],trend_time[...,i*self.v_dim:(i+1)*self.v_dim]) for i in range(self.n_heads))
        trend_time_token_attn_output = torch.cat([t_attn for t_attn in trend_time_token_attn_output],dim=-1)        

        season_token_attn_output = self.season_attn_layer(season_token, season_token, season_token,
                                            attn_mask=season_attn_mask,
                                            key_padding_mask=season_key_padding_mask,
                                            need_weights=True,average_attn_weights=False)
        
        # print('v_dim',v_dim,token_attn_output[1][:,0,...].shape,x_time.shape)
        season_time_token_attn_output = (torch.einsum('bll,blt->blt',season_token_attn_output[1][:,i,...],season_time[...,i*self.v_dim:(i+1)*self.v_dim]) for i in range(self.n_heads))
        season_time_token_attn_output = torch.cat([t_attn for t_attn in season_time_token_attn_output],dim=-1)   
        
        
        season_cross_attn_output = self.cross_attn_layer1(trend_time_token_attn_output, season_time_token_attn_output, season_time_token_attn_output, 
                                                  attn_mask=None,key_padding_mask=season_key_padding_mask,need_weights=False,average_attn_weights=False)    
        trend_cross_attn_output = self.cross_attn_layer2(season_time_token_attn_output, trend_time_token_attn_output, trend_time_token_attn_output, 
                                                  attn_mask=None,key_padding_mask=trend_key_padding_mask,need_weights=False,average_attn_weights=False)    

        season_time = trend_cross_attn_output[0]+season_time
        trend_time = season_cross_attn_output[0]+trend_time

        return self.dropout1(trend_token_attn_output[0]),self.dropout1(trend_time),self.dropout1(season_token_attn_output[0]),self.dropout1(season_time)   
      
    # feed forward block
    def _ff_block(self, trend_token: Tensor, trend_time: Tensor,season_token: Tensor, season_time: Tensor) -> Tuple[Tensor,Tensor]:
        trend_x = torch.cat([trend_token,trend_time],dim=-1)
        trend_x = self.linear2(self.dropout(self.activation(self.linear1(trend_x))))
        trend_x = self.dropout2(trend_x)
        
        season_x = torch.cat([season_token,season_time],dim=-1)
        season_x = self.linear4(self.dropout(self.activation(self.linear3(season_x))))
        season_x = self.dropout2(season_x)
        return trend_x[...,:trend_token.shape[-1]],trend_x[...,trend_token.shape[-1]:],season_x[...,:season_token.shape[-1]],season_x[...,season_token.shape[-1]:]



class TimeTokenCrossEncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_time: int,nfeature:int,n_heads: int=1,season_emb_in: int=4,trend_emb_in: int=8,n_layers: int=1,trend_max_token_num:int=16,
                 season_max_token_num:int=16,dim_feedforward: int = 512, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,  
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,device=None, dtype=None):
        super(TimeTokenCrossEncoderBlock, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.trend_norm0 = LayerNorm(trend_emb_in, eps=layer_norm_eps, **factory_kwargs)
        self.trend_in_proj = Linear(trend_emb_in, d_model, **factory_kwargs)
        
        self.season_norm0 = LayerNorm(season_emb_in, eps=layer_norm_eps, **factory_kwargs)
        self.season_in_proj = Linear(season_emb_in, d_model, **factory_kwargs)
        
        for i in range(n_layers):
            self.encoder_layers = nn.ModuleList([TimeTokenCrossEncoderLayer(d_model=d_model,d_time=d_time,n_heads=n_heads,dim_feedforward=dim_feedforward,dropout=dropout,
                                                                       activation=activation,layer_norm_eps=layer_norm_eps,batch_first=batch_first,
                                                                       norm_first=norm_first,device=device,dtype=dtype) for _ in range(n_layers)])
            
        self.d_model = d_model
        self.d_time = d_time
        self.n_heads = n_heads
        self.nfeature = nfeature
        self.layer_norm = nn.LayerNorm(d_time)
        self.dropout = nn.Dropout(dropout)
        # self.token_out_proj = Linear(d_model*n_heads, token_out_dim,**factory_kwargs)
        self.trend_token_agg = nn.MaxPool1d(kernel_size=trend_max_token_num, stride=trend_max_token_num)
        self.trend_out_proj = nn.Sequential(Linear(d_model+d_time, d_time,**factory_kwargs),nn.GELU(),self.layer_norm,self.dropout)
        
        self.season_token_agg = nn.MaxPool1d(kernel_size=season_max_token_num, stride=season_max_token_num)
        self.season_out_proj = nn.Sequential(Linear(d_model+d_time, d_time,**factory_kwargs),nn.GELU(),self.layer_norm,self.dropout)


        # print('token agg fun',self.token_aggregate_func)
        
    def forward(self, trend_token: Tensor, trend_time: Tensor, season_token: Tensor, season_time: Tensor, 
                trend_src_mask: Optional[Tensor] = None,season_src_mask: Optional[Tensor] = None,
                trend_src_key_padding_mask: Optional[Tensor] = None,season_src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor,Tensor]:

       
        # x_token = torch.cat([x_token for _ in range(self.n_heads)],dim=-1)
        trend_token = self.trend_in_proj(trend_token) 
        season_token = self.season_in_proj(season_token)
        
        self.trend_time_mean = trend_time.sum(dim=-1,keepdim=True)/(trend_time!=0).sum(dim=-1,keepdim=True)
        self.trend_time_mean[torch.isnan(self.trend_time_mean)] = 0
        self.trend_time_std = (trend_time - self.trend_time_mean).pow(2)
        
        self.trend_time_std = torch.sqrt(self.trend_time_std.sum(dim=-1,keepdim=True)/(trend_time!=0).sum(dim=-1,keepdim=True))
        self.trend_time_std[(self.trend_time_std==0) | torch.isnan(self.trend_time_std)] = 1
        trend_time[trend_time!=0] = ((trend_time - self.trend_time_mean)/self.trend_time_std)[trend_time!=0]  
        # print('check nan',torch.isnan(x_time).sum(),torch.isnan(self.x_time_std).sum(),torch.isnan(self.x_time_mean).sum())
        
        for layer in self.encoder_layers:
            # print('check x',x_token.shape,x_time.shape)
            trend_token,trend_time,season_token,season_time = layer(trend_token,trend_time,season_token,season_time,
                                                                    trend_src_mask, season_src_mask,trend_src_key_padding_mask, season_src_key_padding_mask)
            # print('check x',x_token[0,0,:3],x_time[0,0:3])

        # out = torch.cat([self.token_out_proj(self.recompose(x_token)),self.recompose(x_time)],dim=-1)
        trend_out = self.trend_out_proj(torch.cat([self.trend_recompose(trend_token),self.trend_recompose(trend_time)],dim=-1))
        season_out = self.season_out_proj(torch.cat([self.season_recompose(season_token),self.season_recompose(season_time)],dim=-1))

        return trend_out,  season_out     


    def trend_recompose(self, x):
        ## aggregate the tokens as a whole
        x = self.trend_token_agg(x.permute(0,2,1)).permute(0,2,1)
        return x
    
    def season_recompose(self, x):
        ## aggregate the tokens as a whole
        x = self.season_token_agg(x.permute(0,2,1)).permute(0,2,1)
        return x



class FullSequenceEncoder(nn.Module):
    def __init__(self, factor, d_model, n_heads, d_feedforward, n_layers,output_attention=False, dropout=0.1, activation="relu"):
        super(FullSequenceEncoder, self).__init__()
        
        self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention), d_model, n_heads),
                        d_model,
                        d_feedforward,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(n_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
    def forward(self, x, attn_mask=None):
        # print('full encoder input',x.shape)
        return self.encoder(x, attn_mask=attn_mask)
    

class FullSequenceDecoder(nn.Module):
    def __init__(self,d_in,d_hidden,pred_len, n_layers=1, dropout=0.1, activation="relu"):
        super(FullSequenceDecoder, self).__init__()
        # print(d_time,d_hidden,pred_len,n_components,n_layers,dropout,activation)
        self.activation = self._get_activation_fn(activation)
        if n_layers>1:
            self.decoder_layers = nn.ModuleList([nn.Sequential(Linear(d_in,d_hidden),self.activation)]+\
                        [nn.Sequential(Linear(d_hidden, d_hidden),self.activation) for _ in range(n_layers-2)]+\
                        [Linear(d_hidden,pred_len)])
        else:
            self.decoder_layers = nn.ModuleList([Linear(d_in,pred_len)])
        
        self.pred_len = pred_len
    
    def forward(self, x):
        # print('full decoder input',x.shape)
        for layer in self.decoder_layers:
            x = layer(x)

        # x = x.reshape(x.shape[0],-1,self.n_components,self.pred_len)
        return x 
    def _get_activation_fn(self,activation: str) -> Callable[[Tensor], Tensor]:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
    
 
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:,:, :x.size(2)]       
        
class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model
        self.d_time = configs.seq_len
        self.n_features = configs.enc_in
        self.components = ['trend','season','residual']
        self.n_heads = configs.n_heads
        self.cp_heads = configs.cp_heads
        self.cp_d_model = configs.cp_d_model
        self.dec_in = configs.dec_in
        self.d_ff = configs.d_ff
        self.cp_d_ff = configs.cp_d_ff
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.te_layers = configs.te_layers
        self.dropout = configs.dropout
        self.activation = configs.activation
        self.factor = configs.factor
        self.token_aggregate_func = configs.token_aggregate_func
        # self.component_encoders = nn.ModuleDict()
        self.component_position_encoders={}
        self.d_hidden = configs.p_hidden_dims
        self.time_embed_type = configs.embed
        self.freq = configs.freq
        time_dim = 4
        
        if self.task_name == 'anomaly_detection':
            self.pred_len = configs.seq_len
        
        self.component_encoder = TimeTokenCrossEncoderBlock(d_model=self.cp_d_model,d_time=self.d_time,nfeature=self.n_features,n_heads=self.cp_heads,n_layers=self.te_layers,season_emb_in=season_token_dim,trend_emb_in=trend_token_dim,
                                                 trend_max_token_num=configs.max_trend_num,season_max_token_num=configs.max_freq_num,dim_feedforward=self.cp_d_ff,dropout=self.dropout,activation=self.activation)
        
        
        tc_dim = 3*self.d_time + time_dim 
        self.emb = Linear(tc_dim,self.d_model)
        self.emb_dropout = nn.Dropout(p=self.dropout)
        self.full_encoder = FullSequenceEncoder(factor=self.factor,d_model=self.d_model,n_heads=self.n_heads,d_feedforward=self.d_ff,
                                                n_layers=self.e_layers,output_attention=self.output_attention,dropout=self.dropout,activation=self.activation)
        
        self.decoder = FullSequenceDecoder(d_in=self.d_model,d_hidden=self.d_ff,pred_len=self.pred_len,n_layers=self.d_layers,
                                          dropout=self.dropout,activation=self.activation)

    def forecast(self, trend_enc,season_enc,x,x_stamp=None):
        # print('check input',torch.where(torch.isnan(trend_enc)),torch.where(torch.isnan(season_enc).sum()))   
        means = x.mean(-1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False)).detach()
        stdev[stdev==0] = 1
        x /= stdev
        
        if x_stamp is not None:
            x_stamp = x_stamp.unsqueeze(1).tile(self.n_features,1)
            x_full = torch.cat([x,trend_enc,season_enc,x_stamp],dim=-1) 
        else:
            x_full = torch.cat([x,trend_enc,season_enc],dim=-1)
                
        # trend_enc = trend_enc.unsqueeze(2)
        # season_enc = season_enc.unsqueeze(2)
        # x = x.unsqueeze(2)
        # x_full = torch.cat([x,trend_enc,season_enc],dim=2)
        # x_full = x_full.reshape(x_full.shape[0],-1,x_full.shape[-1])
        
        # if x_stamp is not None:
        #     x_stamp = x_stamp.permute(0,2,1)
        #     x_full = torch.cat([x_full,x_stamp],dim=1)    
           
        x_full = self.emb_dropout(self.emb(x_full))
        
        enc_out,enc_attn = self.full_encoder(x_full)
        # print('check enc_out',torch.isnan(enc_out).sum())
        dec_out = self.decoder(enc_out)
        # print('check dec_out',torch.isnan(dec_out).sum())
        # dec_out = dec_out[:,:self.n_features*3,:]
        # dec_out = dec_out.reshape(dec_out.shape[0],self.n_features,3,-1)
        # dec_out = dec_out[:,:,0,:]
        
        dec_out = dec_out * stdev
        dec_out = dec_out + means
        # print('check dec_out after',torch.isnan(dec_out).sum())

        return dec_out,enc_attn
    
    def anomaly_detection(self, trend_enc,season_enc,x):    
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # Embedding
        trend_enc = trend_enc.unsqueeze(2)
        season_enc = season_enc.unsqueeze(2)
        x = x.unsqueeze(2)
        x_full = torch.cat([x,trend_enc,season_enc],dim=2)
        x_full = x_full.reshape(x_full.shape[0],-1,x_full.shape[-1])
        x_full = self.emb_dropout(self.emb(x_full))
        
        enc_out,enc_attn = self.full_encoder(x_full)
        dec_out = self.decoder(enc_out)
        
        dec_out = dec_out[:,:self.n_features*3,:]
        dec_out = dec_out.reshape(dec_out.shape[0],self.n_features,3,-1)
        dec_out = dec_out[:,:,0,:]
        # print('dec_out',dec_out.shape,x.shape)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :self.n_features]
        # print('after dec_out',dec_out.shape,stdev.shape)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev
        dec_out = dec_out + means
        return dec_out,enc_attn
    
    
    def forward(self, trend_token,trend_time,season_token,season_time,x,x_stamp=None):
        ## each component is encoded separately, and has shape [batch size, feature number, time length]
        # res = x - trend_time.sum(dim=2) - season_time.sum(dim=2)
        # print('check raw input',torch.isnan(trend_token).sum(),torch.isnan(trend_time).sum(),torch.isnan(season_token).sum(),torch.isnan(season_time).sum(),torch.isnan(x).sum())
                
        trend_mask = trend_token.sum(dim=-1)==0
        trend_mask = trend_mask.reshape(trend_mask.shape[0],-1)
                    
        trend_token = trend_token.reshape(trend_token.shape[0],-1,trend_token.shape[-1])
        trend_time = trend_time.reshape(trend_time.shape[0],-1,trend_time.shape[-1])
        
        season_mask = season_token.sum(dim=-1)==0
        season_mask = season_mask.reshape(season_mask.shape[0],-1)
                    
        season_token = season_token.reshape(season_token.shape[0],-1,season_token.shape[-1])
        season_time = season_time.reshape(season_time.shape[0],-1,season_time.shape[-1])
        
        trend_enc,season_enc = self.component_encoder(trend_token,trend_time,season_token,season_time, 
                                                      trend_src_key_padding_mask=trend_mask,season_src_key_padding_mask=season_mask)
        
        
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            return self.forecast(trend_enc,season_enc,x,x_stamp)
        elif self.task_name == "anomaly_detection":
            return self.anomaly_detection(trend_enc,season_enc,x)
        
        
    

