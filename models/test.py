import unittest
import torch
import torch.nn.functional as F
from TimeTokenFormer import *
import argparse


class TestTimeTokenFormer(unittest.TestCase):

    def setUp(self):
        # Set up any necessary variables or objects here
        self.d_model = 3
        self.n_heads = 4
        self.d_time = 72
        self.d_feedforward = 256
        self.dropout = 0.1
        self.activation = "relu"
        self.layer_norm_eps = 1e-5
        self.batch_first = True
        self.norm_first = False
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.d_hidden=64
        self.pred_len=20
        self.n_components=3
        self.n_features = 2
        self.factor = 5
        self.n_layers = 2
        self.output_attention = False
        # print(self.d_hidden,self.pred_len,self.n_components)
        self.encoder_layer = TimeTokenEncoderLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=self.d_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=self.batch_first,
            norm_first=self.norm_first,
            device=self.device,
            dtype=self.dtype
        )

        self.full_sequence_encoder = FullSequenceEncoder(
            factor=self.factor,
            d_time=self.d_time,
            n_heads=self.n_heads,
            d_feedforward=self.d_feedforward,
            n_layers=self.n_layers,
            output_attention=self.output_attention,
            dropout=self.dropout,
            activation="relu"
        )
        
        self.full_sequence_decoder = FullSequenceDecoder(
            d_time=self.d_time,
            d_hidden=self.d_hidden,
            pred_len=self.pred_len,
            n_components=self.n_components,
            n_layers=2,
            dropout=self.dropout,
            activation="relu"
        )
        self.components = ['trend', 'season', 'outlier']
        self.component_encoders={}
        self.token_aggregate_func = 'mean'
        for component in self.components:
            if component == 'trend':
                self.component_encoders[component] = TimeTokenEncoderBlock(d_model=self.d_model,d_time=self.d_time,nfeature=self.n_features,n_heads=self.n_heads,n_layers=self.n_layers,
                                                     restore_time_func=trend_token_restore_time_func,token_aggregate_func=self.token_aggregate_func,
                                                     dim_feedforward=self.d_feedforward,dropout=self.dropout,activation=self.activation)
            elif component == 'season':
                self.component_encoders[component] = TimeTokenEncoderBlock(d_model=self.d_model,d_time=self.d_time,nfeature=self.n_features,n_heads=self.n_heads,n_layers=self.n_layers,
                                                     restore_time_func=season_token_restore_time_func,token_aggregate_func=self.token_aggregate_func,
                                                     dim_feedforward=self.d_feedforward,dropout=self.dropout,activation=self.activation)
            elif component == "outlier" or component == "residual":
                self.component_encoders[component] = FullSequenceEncoder(factor=self.factor,d_time=self.d_time,n_heads=self.n_heads,d_feedforward=self.d_feedforward,
                                                               n_layers=self.n_layers,output_attention=self.output_attention,dropout=self.dropout,activation=self.activation)
        
        parser = argparse.ArgumentParser(description='TimesNet')
        
        args = parser.parse_args()
        args.task_name = 'long_term_forecasting'
        args.d_model = self.d_model
        args.n_heads = self.n_heads
        args.d_ff = self.d_feedforward
        args.d_time = self.d_time
        args.dropout = self.dropout
        args.activation = self.activation
        args.layer_norm_eps = self.layer_norm_eps
        args.batch_first = self.batch_first
        args.norm_first = self.norm_first
        args.device = self.device
        args.dtype = self.dtype
        args.d_hidden = self.d_hidden
        args.pred_len = self.pred_len
        args.n_components = self.n_components
        args.n_features = self.n_features
        args.factor = self.factor
        args.n_layers = self.n_layers
        args.output_attention = self.output_attention
        args.token_aggregate_func = self.token_aggregate_func
        args.components = self.components
        
            
        
        self.model = Model(args)
        
        
    def test_time_token_former_forward(self):
        x_tokens_dict = {'trend':torch.rand(10, 24, self.d_model), 'season':torch.rand(10, 24, self.d_model), 'outlier':torch.rand(10, self.n_features, self.d_time)}
        output,output_decomp = self.model.forward(x_tokens_dict)
        print(output.shape,output_decomp.shape)
        self.assertEqual(output_decomp.shape, (10, self.n_features, 3, self.pred_len))

    def test_encoder_layer_forward(self):
        # Test the forward pass of the encoder layer
        src = torch.rand(10, 32, self.d_model)  # (batch_size, seq_length, d_model)
        src_mask = torch.rand(10, 32, 32) > 0.5  # Random mask
        output = self.encoder_layer(src, src_mask)
        self.assertEqual(output.shape, (10, 32, self.d_model * self.n_heads))

    def test_encoder_layer_initialization(self):
        # Test the initialization of the encoder layer
        self.assertEqual(len(self.encoder_layer.attn_heads), self.n_heads)
        self.assertIsInstance(self.encoder_layer.linear1, torch.nn.Linear)
        self.assertIsInstance(self.encoder_layer.linear2, torch.nn.Linear)
        self.assertIsInstance(self.encoder_layer.norm1, torch.nn.LayerNorm)
        self.assertIsInstance(self.encoder_layer.norm2, torch.nn.LayerNorm)

    def test_full_sequence_encoder_forward(self):
        # Test the forward pass of the full sequence encoder
        x = torch.rand(10, 32, self.d_time)  # (batch_size, seq_length, d_time)
        output,attn = self.full_sequence_encoder(x)
        self.assertEqual(output.shape, (10, 32, self.d_time))
        
    def test_token_encoder_block_forward(self):
        # Test the forward pass of the token encoder block
        x = torch.rand(10, 24, 72)
        x[...,0] = torch.randint(low=2,high=72,size=(10,24))
        output,_ = self.component_encoders['outlier'](x)
        self.assertEqual(output.shape, (10, 24, self.d_time))
        
    def test_full_sequence_decoder_forward(self):
        # Test the forward pass of the full sequence decoder
        x = torch.rand(10, 24, self.d_time)  # (batch_size, seq_length, d_time)
        output = self.full_sequence_decoder(x)
        self.assertEqual(output.shape, (10, 8, 3, self.pred_len))

    def testget_aggregate_func(self):
        # Test the get_aggregate_func function
        self.assertEqual(get_aggregate_func('sum'), torch.sum)
        self.assertEqual(get_aggregate_func('mean'), torch.mean)
        self.assertEqual(get_aggregate_func('max'), torch.max)
        self.assertEqual(get_aggregate_func('min'), torch.min)
        with self.assertRaises(ValueError):
            get_aggregate_func('invalid')

if __name__ == '__main__':
    unittest.main()