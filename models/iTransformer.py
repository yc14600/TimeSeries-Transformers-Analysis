import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.no_skip = configs.no_skip
        print('no_skip',self.no_skip)
        self.fuse_decoder = configs.fuse_decoder
        self.decoder_type = configs.decoder_type
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    no_skip = self.no_skip
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            
            if configs.fuse_decoder:
                print('add a fuse layer of decoder')
                # self.projection = nn.Linear(configs.d_model * (4+configs.enc_in),configs.pred_len * (4+configs.enc_in))
                if configs.decoder_type == 'conv2d':
                    kw = 8
                    self.fuse_proj = nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=(4+configs.enc_in,kw),
                        padding='same'
                        # groups=1  # Ensure all channels are fused together
                    )
                elif configs.decoder_type == 'MLP':
                    self.fuse_proj = nn.Sequential(nn.Linear(configs.d_model * (4+configs.enc_in),configs.d_model * (4+configs.enc_in),bias=True),nn.ReLU())
             
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
            
    def get_attention(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # print('enc_out 0',enc_out.shape)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return attns
    

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # print('x_enc 0',x_enc.shape,x_mark_enc.shape)
        if self.decoder_type != 'noNorm':
            
            means = x_enc.mean(1, keepdim=True).detach()
            # print('means',means.shape)  
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            # print('stdev',stdev.shape)  
            x_enc = x_enc / stdev
        # print('x_enc 1',x_enc.shape,x_mark_enc.shape)   
        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # print('enc_out 0',enc_out.shape)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print('enc_out 1',enc_out.shape)
        if self.fuse_decoder:
            # s1,s2,s3 = enc_out.shape
            # enc_out = enc_out.view(s1,s2*s3)
            # # print('enc_out flat',enc_out.shape)
            # flat_dec_out = self.projection(enc_out)
            # dec_out = flat_dec_out.reshape(s1,s2,-1)
            if self.decoder_type == 'conv2d':
                enc_out = enc_out.unsqueeze(1)
                
                enc_out = self.fuse_proj(enc_out)
                enc_out = enc_out.squeeze(1)
            elif self.decoder_type == 'MLP':
                s1,s2,s3 = enc_out.shape
                enc_out = enc_out.view(s1,s2*s3)
                # print('enc_out flat',enc_out.shape)
                flat_enc_out = self.fuse_proj(enc_out)
                enc_out = flat_enc_out.reshape(s1,s2,-1)

        dec_out = self.projection(enc_out)           
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        if self.decoder_type != 'noNorm':
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        print('x_enc',x_enc.shape)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        print('enc_out',enc_out.shape)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        print('dec_out',dec_out.shape)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        print('std',stdev.shape)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.output_attention:
                dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :], attns
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
    
    def latent_rep(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return enc_out
