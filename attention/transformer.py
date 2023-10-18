import torch
from torch import nn
from attention.mha import MultiheadAttention
from layers import ComplexLinear, Complex_Dropout, Complex_LayerNorm, Complex_Conv1d, Complex_BatchNorm1d
# from layers import Complex_LayerNorm_naiv as Complex_LayerNorm
from complex_activations import CReLU
from attention.positional_encoding import positional_encoding


class Transformer(nn.Module):  # Transformer for sequence generation
    def __init__(self, input_dim, output_dim, embed_dim, num_heads, layers, attn_enc_dropout=0.1, attn_src_dropout=0.1, attn_tgt_dropout=0.1, relu_dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.pos_enc = positional_encoding(embed_dim=embed_dim)

        self.in_embedding = ComplexLinear(input_dim, embed_dim)
        self.out_embedding = ComplexLinear(output_dim, embed_dim)

        self.output_linear = ComplexLinear(embed_dim, output_dim)

        self.encoder = TransformerEncoder(embed_dim, num_heads, layers, attn_dropout=attn_enc_dropout, relu_dropout=relu_dropout)
        self.decoder = TransformerDecoder(embed_dim, layers, num_heads=num_heads, src_attn_dropout=attn_src_dropout, tgt_attn_dropout=attn_tgt_dropout, relu_dropout=relu_dropout, tgt_mask=True)

    def forward(self, src, tgt):
        batch_size, tokens, _ = src.shape
        src_pos = self.pos_enc(torch.arange(tokens))
        tgt_pos = self.pos_enc(torch.arange(start=tokens, end=tokens + tgt.shape[1]))

        src = self.in_embedding(src) + src_pos.type(torch.complex64)
        tgt = self.out_embedding(tgt) + tgt_pos.type(torch.complex64)
        src = self.encoder(src)
        tgt = self.decoder(tgt, src)

        tgt = self.output_linear(tgt)

        return tgt


class Transformer_Pred(nn.Module):  # Transformer for Prediction (Encoder only)
    def __init__(self, input_dim, output_dim, embed_dim, hidden_dim, num_heads, layers, sm_variante='real', attn_enc_dropout=0.1, relu_dropout=0.1, out_dropout=0.1, tokens=None, device='cuda'):
        super().__init__()
        self.name = 'CTrans_Pred'
        self.device = device
        self.tokens = tokens
        self.sm_variante = sm_variante

        self.conv = nn.Sequential(
                Complex_Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
                Complex_BatchNorm1d(num_channel=16),
                CReLU(),

                Complex_Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
                Complex_BatchNorm1d(num_channel=32),
                CReLU(),

                Complex_Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
                Complex_BatchNorm1d(num_channel=64),
                CReLU(),

                Complex_Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
                Complex_BatchNorm1d(num_channel=64),
                CReLU(),

                Complex_Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
                Complex_BatchNorm1d(num_channel=128),
                CReLU(),
                nn.Flatten())

        channels = (((((input_dim - 6) // 2 + 1 - 3) // 2 + 1 - 3) // 2 + 1 - 3) // 2 + 1 - 3) // 2 + 1  # calculate num channels after self.conv due to stride

        self.embed_dim = embed_dim

        self.pos_enc = positional_encoding(embed_dim=embed_dim)

        input_dim_trans = channels * 128
        self.in_embedding = ComplexLinear(input_dim_trans, embed_dim)
        self.out_dropout = Complex_Dropout(out_dropout)

        self.output_linear_1 = ComplexLinear(embed_dim, hidden_dim)
        self.final_out = nn.Linear(2, 1)
        self.relu = CReLU()

        self.output_linear_2 = ComplexLinear(hidden_dim, output_dim)

        self.encoder = TransformerEncoder(embed_dim, num_heads, layers, sm_variante=self.sm_variante, attn_dropout=attn_enc_dropout, relu_dropout=relu_dropout)
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

        src = self.in_embedding(src) + src_pos.type(torch.complex64)
        src = self.encoder(src)

        src = self.output_linear_2(self.out_dropout(self.relu(self.output_linear_1(src))))

        return torch.squeeze(self.final_out(torch.view_as_real(src)))


class Transformer_Generation(nn.Module):  # Transformer for Prediction (Encoder only)
    def __init__(self, input_dim, output_dim, embed_dim, hidden_dim, num_heads, layers, sm_variante='real', attn_enc_dropout=0.1, src_attn_dropout=0.1, tgt_attn_dropout=0.1, relu_dropout=0.1, out_dropout=0.1, src_tokens=None, tgt_tokens=None, device='cuda'):
        super().__init__()
        self.name = 'CTrans_gen'
        self.device = device
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        self.sm_variante = sm_variante

        self.conv_src = nn.Sequential(
                Complex_Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
                Complex_BatchNorm1d(num_channel=16),
                CReLU(),

                Complex_Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
                Complex_BatchNorm1d(num_channel=32),
                CReLU(),

                Complex_Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
                Complex_BatchNorm1d(num_channel=64),
                CReLU(),

                Complex_Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
                Complex_BatchNorm1d(num_channel=64),
                CReLU(),

                Complex_Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
                Complex_BatchNorm1d(num_channel=128),
                CReLU(),
                nn.Flatten())

        # self.conv_tgt = nn.Sequential(
        #         Complex_Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
        #         Complex_BatchNorm1d(num_channel=16),
        #         CReLU(),

        #         Complex_Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
        #         Complex_BatchNorm1d(num_channel=32),
        #         CReLU(),

        #         Complex_Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
        #         Complex_BatchNorm1d(num_channel=64),
        #         CReLU(),

        #         Complex_Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
        #         Complex_BatchNorm1d(num_channel=64),
        #         CReLU(),

        #         Complex_Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
        #         Complex_BatchNorm1d(num_channel=128),
        #         CReLU(),
        #         nn.Flatten())

        channels = (((((input_dim - 6) // 2 + 1 - 3) // 2 + 1 - 3) // 2 + 1 - 3) // 2 + 1 - 3) // 2 + 1  # calculate num channels after self.conv due to stride

        self.embed_dim = embed_dim

        self.pos_enc = positional_encoding(embed_dim=embed_dim)

        input_dim_trans = channels * 128
        self.in_embedding_src = ComplexLinear(input_dim_trans, embed_dim)
        self.in_embedding_tgt = ComplexLinear(128, embed_dim)
        self.out_dropout = Complex_Dropout(out_dropout)

        self.output_linear_1 = ComplexLinear(embed_dim, hidden_dim)
        self.final_out = nn.Linear(2, 1)
        self.relu = CReLU()

        self.output_linear_2 = ComplexLinear(hidden_dim, output_dim)

        self.encoder = TransformerEncoder(embed_dim, num_heads, layers, sm_variante=self.sm_variante, attn_dropout=attn_enc_dropout, relu_dropout=relu_dropout)
        self.decoder = TransformerDecoder(embed_dim, num_heads, layers, sm_variante=self.sm_variante, src_attn_dropout=src_attn_dropout, tgt_attn_dropout=tgt_attn_dropout, relu_dropout=relu_dropout, tgt_mask=True)

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

        src = self.in_embedding_src(src) + src_pos.type(torch.complex64)
        src = self.encoder(src)

        if tgt is not None:
            batch_size, tgt_tokens, features = tgt.shape

            tgt = tgt[:, :-1]  # remove last token for right shift

            sos = torch.zeros([batch_size, 1, features], dtype=torch.complex64, device=self.device)  # start of sequence token
            tgt = torch.cat([sos, tgt], dim=1)

            if self.tgt_tokens is None:  # if tokens have different length
                tgt_pos = self.pos_enc(torch.arange(tgt_tokens, device=self.device))
            else:  # if tokens have all the same length
                tgt_pos = self.tgt_pos

            tgt = self.in_embedding_tgt(tgt) + tgt_pos.type(torch.complex64)
            out = self.decoder(tgt, src)

        out = self.output_linear_2(self.out_dropout(self.relu(self.output_linear_1(out))))

        return torch.squeeze(self.final_out(torch.view_as_real(out)))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, sm_variante='real', attn_dropout=0.1, relu_dropout=0.1):
        super().__init__()
        self.sm_variante = sm_variante
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    sm_variante=self.sm_variante,
                                    attn_dropout=attn_dropout,
                                    relu_dropout=relu_dropout)
            for _ in range(layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads=4, sm_variante='real', attn_dropout=0.1, relu_dropout=0.1):
        # no mask, since no necessarity (except for potential padding mask, not needed here)
        super().__init__()
        self.embed_dim = embed_dim
        self.sm_variante = sm_variante
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout,
            bias=True,
            sm_variante=self.sm_variante
        )

        self.relu = CReLU()

        self.relu_dropout = Complex_Dropout(relu_dropout)

        self.linear1 = ComplexLinear(self.embed_dim, self.embed_dim)
        self.linear2 = ComplexLinear(self.embed_dim, self.embed_dim)

        self.layer_norms = nn.ModuleList([Complex_LayerNorm(embed_dim) for _ in range(2)])

    def forward(self, x):
        residual = x

        x, _ = self.self_attn(x, x, x)

        x += residual

        x = self.layer_norms[0](x)

        residual = x

        x = self.linear1(x)

        x = self.relu(x)

        x = self.relu_dropout(x)

        x = self.linear2(x)

        x += residual

        x = self.layer_norms[1](x)

        return x


class TransformerDecoder(nn.Module):

    def __init__(self, embed_dim, num_heads, num_layers, sm_variante='real', src_attn_dropout=0.1, tgt_attn_dropout=0.1, relu_dropout=0.1, tgt_mask=True):
        super().__init__()
        self.sm_variante = sm_variante

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.tgt_mask = tgt_mask

        self.src_attn_dropout = src_attn_dropout
        self.tgt_attn_dropout = tgt_attn_dropout
        self.relu_dropout = relu_dropout

        self.layers = nn.ModuleList([TransformerDecoderLayer(self.embed_dim,
                                     num_heads=self.num_heads,
                                     sm_variante=self.sm_variante,
                                     src_attn_dropout=self.src_attn_dropout,
                                     tgt_attn_dropout=self.tgt_attn_dropout,
                                     relu_dropout=self.relu_dropout,
                                     tgt_mask=self.tgt_mask) for _ in range(num_layers)])

    def forward(self, x, enc):
        for layer in self.layers:
            x = layer(x, enc)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, sm_variante='real', src_attn_dropout=0.1, tgt_attn_dropout=0.1, relu_dropout=0.1, tgt_mask=False):
        super().__init__()
        self.sm_variante = sm_variante

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=src_attn_dropout,
            sm_variante=sm_variante,
            bias=True
        )

        self.tgt_mask = tgt_mask

        self.attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=tgt_attn_dropout,
            sm_variante=sm_variante,
            bias=True
        )

        self.linear1 = ComplexLinear(self.embed_dim, self.embed_dim)
        self.linear2 = ComplexLinear(self.embed_dim, self.embed_dim)
        self.relu = CReLU()
        self.relu_dropout = Complex_Dropout(relu_dropout)  # add size for performance

        self.layer_norms = nn.ModuleList([Complex_LayerNorm(embed_dim) for _ in range(3)])

    def forward(self, x, enc):

        residual = x

        if self.tgt_mask:
            attn_mask = self.generate_square_subsequent_mask(x.shape[1])
            x, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        else:
            x, _ = self.self_attn(x, x, x)

        x += residual
        x = self.layer_norms[0](x)
        residual = x

        x, _ = self.attn(x, enc, enc)

        x += residual
        x = self.layer_norms[1](x)
        residual = x

        x = self.linear1(x)
        x = self.relu(x)
        x = self.relu_dropout(x)
        x = self.linear2(x)

        x += residual
        x = self.layer_norms[2](x)

        return x

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return torch.complex(mask, mask)
