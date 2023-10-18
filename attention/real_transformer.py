import torch
from torch import nn
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from attention.positional_encoding import positional_encoding

class real_Transformer_Pred(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, hidden_dim, num_heads, layers, attn_enc_dropout=0.1, relu_dropout=0.1, out_dropout=0.1, tokens=None, device='cuda'):
        super().__init__()
        self.name = 'RTrans_Pred'
        self.device = device
        self.tokens = tokens

        self.conv = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
                nn.BatchNorm1d(num_features=16),
                nn.ReLU(),

                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
                nn.BatchNorm1d(num_features=32),
                nn.ReLU(),

                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
                nn.BatchNorm1d(num_features=64),
                nn.ReLU(),

                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
                nn.BatchNorm1d(num_features=64),
                nn.ReLU(),

                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
                nn.BatchNorm1d(num_features=128),
                nn.ReLU(),
                nn.Flatten())

        channels = (((((input_dim * 2 - 6) // 2 + 1 - 3) // 2 + 1 - 3) // 2 + 1 - 3) // 2 + 1 - 3) // 2 + 1  # calculate num channels after self.conv due to stride

        self.embed_dim = embed_dim

        self.pos_enc = positional_encoding(embed_dim=embed_dim)

        input_dim_trans = channels * 128
        self.in_embedding = nn.Linear(input_dim_trans, embed_dim)
        self.out_dropout = nn.Dropout(out_dropout)

        self.output_linear_1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.output_linear_2 = nn.Linear(hidden_dim, output_dim)

        self.encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.encoder = TransformerEncoder(self.encoder_layer, layers)
        if self.tokens is not None:  # if tokens always habe the same length
            self.src_pos = self.pos_enc(torch.arange(self.tokens, device=self.device))

    def forward(self, src):
        batch_size, tokens, features = src.shape
        src = src.view(-1, 1, features)

        src = self.conv(src)

        src = torch.reshape(src, [batch_size, tokens, -1])

        if self.tokens is None:  # if tokens have different length
            src_pos = self.pos_enc(torch.arange(tokens, device=self.device))
        else:  # if tokens have all the same length
            src_pos = self.src_pos

        src = self.in_embedding(src) + src_pos
        src = self.encoder(src)

        src = self.output_linear_2(self.out_dropout(self.relu(self.output_linear_1(src))))

        return src  # torch.squeeze(self.final_out(torch.view_as_real(src)))


class real_Transformer_Gen(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, hidden_dim, num_heads, layers, attn_enc_dropout=0.1, src_attn_dropout=0.1, tgt_attn_dropout=0.1, relu_dropout=0.1, out_dropout=0.1, src_tokens=None, tgt_tokens=None, device='cuda'):
        super().__init__()
        self.name = 'real_Trans_gen'
        self.device = device
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens

        self.conv_src = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
                nn.BatchNorm1d(num_features=16),
                nn.ReLU(),

                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
                nn.BatchNorm1d(num_features=32),
                nn.ReLU(),

                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
                nn.BatchNorm1d(num_features=64),
                nn.ReLU(),

                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
                nn.BatchNorm1d(num_features=64),
                nn.ReLU(),

                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
                nn.BatchNorm1d(num_features=128),
                nn.ReLU(),
                nn.Flatten())

        # self.conv_tgt = nn.Sequential(
        #         nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
        #         nn.BatchNorm1d(num_features=16),
        #         nn.ReLU(),

        #         nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
        #         nn.BatchNorm1d(num_features=32),
        #         nn.ReLU(),

        #         nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
        #         nn.BatchNorm1d(num_features=64),
        #         nn.ReLU(),

        #         nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
        #         nn.BatchNorm1d(num_features=64),
        #         nn.ReLU(),

        #         nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
        #         nn.BatchNorm1d(num_features=128),
        #         nn.ReLU(),
        #         nn.Flatten())

        channels = (((((input_dim * 2 - 6) // 2 + 1 - 3) // 2 + 1 - 3) // 2 + 1 - 3) // 2 + 1 - 3) // 2 + 1  # calculate num channels after self.conv due to stride

        self.embed_dim = embed_dim

        self.pos_enc = positional_encoding(embed_dim=embed_dim)

        input_dim_trans = channels * 128
        self.in_embedding_src = nn.Linear(input_dim_trans, embed_dim)
        self.in_embedding_tgt = nn.Linear(128, embed_dim)
        self.out_dropout = nn.Dropout(out_dropout)

        self.output_linear_1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.output_linear_2 = nn.Linear(hidden_dim, output_dim)

        self.encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.encoder = TransformerEncoder(self.encoder_layer, layers)
        self.decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.decoder = TransformerDecoder(self.decoder_layer, layers)

        if self.src_tokens is not None:  # if tokens always have the same length
            self.src_pos = self.pos_enc(torch.arange(self.src_tokens, device=self.device))
        if self.tgt_tokens is not None:  # if tokens always have the same length
            self.tgt_pos = self.pos_enc(torch.arange(self.tgt_tokens, device=self.device))

    def forward(self, src, tgt=None, length=None):
        batch_size, src_tokens, features = src.shape
        src = src.view(-1, 1, features)

        src = self.conv_src(src)

        src = torch.reshape(src, [batch_size, src_tokens, -1])

        if self.src_tokens is None:  # if tokens have different length
            src_pos = self.pos_enc(torch.arange(src_tokens, device=self.device))
        else:  # if tokens have all the same length
            src_pos = self.src_pos

        src = self.in_embedding_src(src)
        src = src + src_pos
        src = self.encoder(src)

        if tgt is not None:
            batch_size, tgt_tokens, features = tgt.shape

            tgt = tgt[:, :-1]  # remove last token for right shift

            sos = torch.zeros([batch_size, 1, features], device=self.device)  # start of sequence token
            tgt = torch.cat([sos, tgt], dim=1)

            if self.tgt_tokens is None:  # if tokens have different length
                tgt_pos = self.pos_enc(torch.arange(tgt_tokens, device=self.device))
            else:  # if tokens have all the same length
                tgt_pos = self.tgt_pos

            tgt = self.in_embedding_tgt(tgt) + tgt_pos
            out = self.decoder(tgt, src)

        out = self.output_linear_2(self.out_dropout(self.relu(self.output_linear_1(out))))

        return out
