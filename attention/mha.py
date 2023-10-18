import torch
from torch import nn
from layers import ComplexLinear, Complex_Dropout
from torch.nn.functional import softmax


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, attn_dropout=0, bias=True, sm_variante='real', device='cuda'):
        super().__init__()
        self.sm_variante = sm_variante
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        self.device = device
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.in_proj_q = ComplexLinear(embed_dim, embed_dim, bias=bias)
        self.in_proj_k = ComplexLinear(embed_dim, embed_dim, bias=bias)
        self.in_proj_v = ComplexLinear(embed_dim, embed_dim, bias=bias)
        self.out_proj = ComplexLinear(embed_dim, embed_dim, bias=bias)

        self.cdropout = Complex_Dropout(attn_dropout)
        self.product = self.sm_variante[-2:]
        self.sm_variante = self.sm_variante[:-2]

    def forward(self, query, key, value, attn_mask=None):

        batch_size, target_len, embed_dim = query.size()
        _, input_len, _ = key.size()

        q = self.in_proj_q(query)
        k = self.in_proj_k(key)
        v = self.in_proj_v(value)

        q = q.transpose(1, 0).contiguous().view(target_len, batch_size * self.num_heads, self.head_dim).transpose(1, 0)
        k = k.transpose(1, 0).contiguous().view(input_len, batch_size * self.num_heads, self.head_dim).transpose(1, 0)
        v = v.transpose(1, 0).contiguous().view(input_len, batch_size * self.num_heads, self.head_dim).transpose(1, 0)

        if self.product == 'cp':
            attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scaling
        elif self.product == 'ip':
            attn_weights = torch.bmm(q, torch.conj_physical(k).transpose(1, 2)) * self.scaling
        else:
            raise ValueError(f'{self.product} is not a valid argument')

        attn_weights = self.softmax_variants(attn_weights, attn_mask=attn_mask, sm_variante=self.sm_variante)
        attn_weights = self.cdropout(attn_weights)
        attn = torch.bmm(attn_weights, v)

        attn = attn.transpose(1, 0).contiguous().view(target_len, batch_size, self.embed_dim).transpose(1, 0)
        attn = self.out_proj(attn)

        return attn, attn_weights

    def softmax_variants(self, input, attn_mask=None, sm_variante='real'):
        if sm_variante == 'real':
            return self.softmax_real(input, attn_mask=attn_mask)
        elif sm_variante == 'abs':
            return self.softmax_abs(input, attn_mask=attn_mask)
        elif sm_variante == 'naiv':
            return self.softmax_naiv(input, attn_mask=attn_mask)
        elif sm_variante == 'absonly':
            return self.softmax_abs_only(input, attn_mask=attn_mask)
        else:
            raise ValueError(f'{sm_variante} is not a valid variant for C-softmax')

    def softmax_abs(self, input, attn_mask=None):
        abso = torch.abs(input)
        if attn_mask is not None:
            abso += attn_mask.unsqueeze(0).real.to(self.device)
        return softmax(abso, dim=-1).type(torch.complex64) * torch.sgn(input)

    def softmax_naiv(self, input, attn_mask=None):
        if attn_mask is not None:
            # input += attn_mask.unsqueeze(0).to(self.device)
            input = torch.complex(input.real + attn_mask.unsqueeze(0).to(self.device).real, input.imag + attn_mask.unsqueeze(0).to(self.device).imag)
        return torch.complex(softmax(input.real, dim=-1), softmax(input.imag, dim=-1))

    def softmax_abs_only(self, input, attn_mask=None):
        abso = torch.abs(input)
        if attn_mask is not None:
            abso += attn_mask.unsqueeze(0).real.to(self.device)
        # abso[abso == float('inf')] = -abso[abso == float('inf')]
        return softmax(abso, dim=-1).type(torch.complex64)

    def softmax_real(self, input, attn_mask=None):
        real = torch.real(input)
        if attn_mask is not None:
            real += attn_mask.unsqueeze(0).real.to(self.device)
        # abso[abso == float('inf')] = -abso[abso == float('inf')]
        return softmax(real, dim=-1).type(torch.complex64)

    def min_max_real(self, input, attn_mask=None):  # attnmask does not work yet
        real = torch.real(input)
        mini = torch.min(real, dim=-1)[0].unsqueeze(-1)
        maxi = torch.max(real, dim=-1)[0].unsqueeze(-1)
        return ((real - mini) / (maxi - mini)).type(torch.complex64)

    def min_max_naiv(self, input, attn_mask=None):  # attnmaks does not work yet
        real = torch.real(input)
        rmini = torch.min(real, dim=-1)
        rmaxi = torch.max(real, dim=-1)
        imag = torch.imag(input)
        imini = torch.min(imag, dim=-1)
        imaxi = torch.max(imag, dim=-1)
        return torch.complex((real - rmini) / (rmaxi - rmini), (imag - imini) / (imaxi - imini))

    def min_max_complex(self, input, attn_mask=None):  # attnmask does not work yet
        mini = torch.take_along_dim(input, torch.argmin(input.real, dim=-1, keepdims=True), dim=-1).unsqueeze(-1)
        pos = input - mini
        maxi = torch.take_along_dim(input, torch.argmax(torch.abs(pos), dim=-1, keepdims=True), dim=-1).unsqueeze(-1)
        return pos / maxi

