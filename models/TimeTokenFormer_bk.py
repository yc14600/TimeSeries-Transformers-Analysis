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


trend_token_dim = 6
season_token_dim = 4
residual_token_dim = 8

# out = self.out_proj(torch.cat([self.recompose(x_token),self.recompose(x_time)],dim=-1))

class TimeTokenEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model: int, d_time: int, n_heads: int = 1,emb_in: int = 4, dim_feedforward: int = 512, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,  
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,device=None, dtype=None)->None:
        nn.Module.__init__(self)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.norm0 = LayerNorm(emb_in, eps=layer_norm_eps, **factory_kwargs)
        # self.dropout0 = Dropout(dropout)
        self.in_proj = nn.Sequential(self.norm0,Linear(emb_in, d_model, **factory_kwargs))
        self.attn_layer = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
        self.v_dim = int(d_time/n_heads)
        
        self.d_model = d_model
        self.d_time = d_time
        self.n_heads = n_heads

        self.dropout = Dropout(dropout)
        self.linear1 = Linear(d_model+d_time, dim_feedforward, **factory_kwargs)
        self.linear2 = Linear(dim_feedforward, d_model+d_time, **factory_kwargs)


        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_time, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm4 = LayerNorm(d_time, eps=layer_norm_eps, **factory_kwargs)
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
    def forward(self, src_token: Tensor, src_time: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor,Tensor]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")

        x_token = src_token
        x_time = src_time
        
        x_token = self.in_proj(x_token)        
        if self.norm_first:
            x_token_o, x_time_o = self._sa_block(self.norm1(x_token),self.norm2(x_time), src_mask, src_key_padding_mask) 
            
            x_token += x_token_o 
            x_time += x_time_o      
            
            x_token_o, x_time_o = self._ff_block(self.norm3(x_token),self.norm4(x_time))
                 
            x_token += x_token_o 
            x_time += x_time_o
            
        else:
            # print('x',x.shape)
            # print('sax',self._sa_block(x, src_mask, src_key_padding_mask).shape)
            x_token_o, x_time_o = self._sa_block(x_token,x_time, src_mask, src_key_padding_mask)
            x_token = self.norm1(x_token_o + x_token)
            x_time = self.norm2(x_time_o + x_time)
            
            x_token_o, x_time_o = self._ff_block(x_token,x_time)
            
            x_token = self.norm3(x_token_o + x_token)
            x_time = self.norm4(x_time_o + x_time)

        return x_token,x_time

    # self-attention block
    def _sa_block(self, x_token: Tensor, x_time:Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tuple[Tensor,Tensor]:
        
        token_attn_output = self.attn_layer(x_token, x_token, x_token,
                                            attn_mask=attn_mask,
                                            key_padding_mask=key_padding_mask,
                                            need_weights=True,average_attn_weights=False)

        
        # print('v_dim',v_dim,token_attn_output[1][:,0,...].shape,x_time.shape)
        time_attn_output = (torch.einsum('bll,blt->blt',token_attn_output[1][:,i,...],x_time[...,i*self.v_dim:(i+1)*self.v_dim]) for i in range(self.n_heads))
        time_attn_output = torch.cat([t_attn for t_attn in time_attn_output],dim=-1)        

        return self.dropout1(token_attn_output[0]),self.dropout1(time_attn_output)  
      
    # feed forward block
    def _ff_block(self, x_token: Tensor, x_time: Tensor) -> Tuple[Tensor,Tensor]:
        x = torch.cat([x_token,x_time],dim=-1)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.dropout2(x)
        return x[...,:x_token.shape[-1]],x[...,x_token.shape[-1]:]


class TimeTokenEncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_time: int,nfeature:int,n_heads: int=1,emb_in: int=4,n_layers: int=1,max_token_num:int=16,
                 dim_feedforward: int = 512, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,  
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,device=None, dtype=None):
        super(TimeTokenEncoderBlock, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        for i in range(n_layers):
            self.encoder_layers = nn.ModuleList([TimeTokenEncoderLayer(d_model=d_model,d_time=d_time,n_heads=n_heads,emb_in=emb_in,dim_feedforward=dim_feedforward,dropout=dropout,
                                                                       activation=activation,layer_norm_eps=layer_norm_eps,batch_first=batch_first,
                                                                       norm_first=norm_first,device=device,dtype=dtype) for _ in range(n_layers)])
        self.d_model = d_model
        self.d_time = d_time
        self.n_heads = n_heads
        self.nfeature = nfeature
        self.layer_norm = nn.LayerNorm(d_time)
        self.dropout = nn.Dropout(dropout)
        # self.token_out_proj = Linear(d_model*n_heads, token_out_dim,**factory_kwargs)
        self.token_agg = nn.MaxPool1d(kernel_size=max_token_num, stride=max_token_num)
        self.out_proj = nn.Sequential(Linear(d_model+d_time, d_time,**factory_kwargs),nn.GELU(),self.layer_norm,self.dropout)

        # print('token agg fun',self.token_aggregate_func)
        
    def forward(self, src_token: Tensor,src_time: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x_token = src_token
        x_time = src_time
        # x_token = torch.cat([x_token for _ in range(self.n_heads)],dim=-1)
        for layer in self.encoder_layers:
            # print('check x',x_token.shape,x_time.shape)
            x_token,x_time = layer(x_token,x_time, src_mask, src_key_padding_mask)
            # print('check x',x_token[0,0,:3],x_time[0,0:3])
            
        # out = torch.cat([self.token_out_proj(self.recompose(x_token)),self.recompose(x_time)],dim=-1)
        out = self.out_proj(torch.cat([self.recompose(x_token),self.recompose(x_time)],dim=-1))
        # out = self.recompose(x_time)
        return out       


    def recompose(self, x):
        # x = x.reshape(x.shape[0],self.nfeature,-1,x.shape[-1])
        # print('recomp x',x.shape)
        ## aggregate the tokens as a whole
        x = self.token_agg(x.permute(0,2,1)).permute(0,2,1)
        # x = torch.sum(x,dim=2) 
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
    def __init__(self,d_in,d_hidden,pred_len,n_components, n_layers=1, dropout=0.1, activation="relu"):
        super(FullSequenceDecoder, self).__init__()
        # print(d_time,d_hidden,pred_len,n_components,n_layers,dropout,activation)
        self.activation = _get_activation_fn(activation)
        if n_layers>1:
            self.decoder_layers = nn.ModuleList([nn.Sequential(Linear(d_in,d_hidden),self.activation)]+\
                        [nn.Sequential(Linear(d_hidden, d_hidden),self.activation) for _ in range(n_layers-2)]+\
                        [Linear(d_hidden,pred_len)])
        else:
            self.decoder_layers = nn.ModuleList([Linear(d_in,pred_len)])
        
        self.n_components = n_components
        self.pred_len = pred_len
    
    def forward(self, x):
        # print('full decoder input',x.shape)
        for layer in self.decoder_layers:
            x = layer(x)

        # x = x.reshape(x.shape[0],-1,self.n_components,self.pred_len)
        return x 
    
 
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
        self.dec_in = configs.dec_in
        self.d_feedforward = configs.d_ff
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.te_layers = configs.te_layers
        self.dropout = configs.dropout
        self.activation = configs.activation
        self.factor = configs.factor
        self.token_aggregate_func = configs.token_aggregate_func
        self.component_encoders = nn.ModuleDict()
        self.component_position_encoders={}
        self.d_hidden = configs.p_hidden_dims
        self.time_embed_type = configs.embed
        self.freq = configs.freq
        
        for component in self.components:
            if component == 'trend':
                # self.component_position_encoders[component] = PositionalEmbedding(d_model=trend_token_dim)
                self.component_encoders[component] = TimeTokenEncoderBlock(d_model=self.d_time,d_time=self.d_time,emb_in=trend_token_dim,nfeature=self.n_features,n_heads=self.cp_heads,n_layers=self.te_layers,
                                                     max_token_num=configs.max_trend_num,dim_feedforward=self.d_feedforward,dropout=self.dropout,activation=self.activation)
            elif component == 'season':
                # self.component_position_encoders[component] = PositionalEmbedding(d_model=season_token_dim)
                self.component_encoders[component] = TimeTokenEncoderBlock(d_model=self.d_time,d_time=self.d_time,emb_in=season_token_dim,nfeature=self.n_features,n_heads=self.cp_heads,n_layers=self.te_layers,
                                                     max_token_num=configs.max_freq_num,dim_feedforward=self.d_feedforward,dropout=self.dropout,activation=self.activation)
            elif component == "residual":
                # self.component_position_encoders[component] = PositionalEmbedding(d_model=residual_token_dim)
                self.component_encoders[component] = TimeTokenEncoderBlock(d_model=self.d_time,d_time=self.d_time,emb_in=residual_token_dim,nfeature=self.n_features,n_heads=self.cp_heads,n_layers=self.te_layers,
                                                     max_token_num=configs.max_trend_num,dim_feedforward=self.d_feedforward,dropout=self.dropout,activation=self.activation)

            else:
                raise ValueError(f'component should be trend, season, outlier, or residual, but got {component}')
        
        tc_dim = 4 * self.d_time + 4 #3*(self.d_time+self.dec_in)+self.d_time
        self.emb = Linear(tc_dim,self.d_model)
        self.emb_dropout = nn.Dropout(p=self.dropout)
        self.full_encoder = FullSequenceEncoder(factor=self.factor,d_model=self.d_model,n_heads=self.n_heads,d_feedforward=self.d_feedforward,
                                                n_layers=self.e_layers,output_attention=self.output_attention,dropout=self.dropout,activation=self.activation)
        
        self.decoder = FullSequenceDecoder(d_in=self.d_model,d_hidden=self.d_model,pred_len=self.pred_len,n_components=len(self.components),n_layers=self.d_layers,
                                          dropout=self.dropout,activation=self.activation)
        # self.time_embed = Linear(self.d_time,tc_dim, bias=False)
    # @profile
    def forward(self, trend_x_token,trend_x_time,seasonal_x_token,seasonal_x_time,residual_x_token,residual_x_time,x,x_stamp, attention_mask=None):
        ## each component is encoded separately, and has shape [batch size, feature number, time length]
        x_component_full = {}
        for component in self.components:
            # print('component',component)
            if component == 'trend':
                input_x_token,input_x_time = trend_x_token, trend_x_time   
            elif component == 'season':
                input_x_token,input_x_time = seasonal_x_token, seasonal_x_time
            elif component == 'residual':
                input_x_token,input_x_time = residual_x_token,residual_x_time 
            # print(input_x_token[0,0,0,0:3],input_x_time[0,0,0,0:3])    
            mask = input_x_token.sum(dim=-1)==0
            mask = mask.reshape(mask.shape[0],-1)
                        
            input_x_token = input_x_token.reshape(input_x_token.shape[0],-1,input_x_token.shape[-1])
            input_x_time = input_x_time.reshape(input_x_time.shape[0],-1,input_x_time.shape[-1])
            x_component_full[component] = self.component_encoders[component](input_x_token,input_x_time,src_key_padding_mask=mask)
            # x_component_full[component] = input_x_time.sum(dim = 2)
            # print('x_component',x_component_full[-1].shape)
        
        x = x.permute(0,2,1)    
        means = x.mean(-1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev
        
        x_stamp = x_stamp.unsqueeze(1).tile(self.n_features,1)
        x_full = torch.cat([x for x in x_component_full.values()]+[x,x_stamp],dim=-1)
        
        x_full = self.emb_dropout(self.emb(x_full))
        
        # print('x',x_full.shape)
        enc_out,enc_attn = self.full_encoder(x_full)
        # print('enc_out',enc_out[0,0,:5])
        dec_out = self.decoder(enc_out)
        # print('dec_out',dec_out[0,0,:5])
        dec_out = dec_out * stdev
        dec_out = dec_out + means
        dec_out = dec_out.permute(0,2,1)
        # if self.output_attention:
        #     return dec_out,enc_attn
        return dec_out,None
    

