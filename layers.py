import torch.nn as nn
import torch


class ComplexLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.Linear_real = torch.nn.Linear(in_features, out_features, bias=bias)
        self.Linear_img = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        real_real = self.Linear_real(input.real)
        img_real = self.Linear_img(input.real)
        real_img = self.Linear_real(input.imag)
        img_img = self.Linear_img(input.imag)
        return real_real - img_img + 1j * (real_img + img_real)  # torch.complex(real_real - img_img, real_img + img_real) not used because of increased memory requirements


class Complex_Conv1d(nn.Module):
    """
    Custom Complex convlayer
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            device=None,
            dtype=None):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype

        self.Conv_real = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            padding_mode=self.padding_mode)

        self.Conv_imag = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            padding_mode=self.padding_mode)

    def forward(self, input):
        out_real = self.Conv_real(input.real) - self.Conv_imag(input.imag)
        out_imag = self.Conv_real(input.imag) + self.Conv_imag(input.real)
        return out_real + 1j * out_imag  # torch.complex(out_real, out_imag) not used because of memory requirements


class Complex_Conv2d(nn.Module):
    """
    Custom Complex convlayer.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            device=None,
            dtype=None):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype

        self.Conv_real = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            padding_mode=self.padding_mode)

        self.Conv_imag = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            padding_mode=self.padding_mode)

    def forward(self, input):
        out_real = self.Conv_real(input.real) - self.Conv_imag(input.imag)
        out_imag = self.Conv_real(input.imag) + self.Conv_imag(input.real)
        return out_real + 1j * out_imag  # torch.complex(out_real, out_imag) not used because of memory requirements


class Complex_BatchNorm2d_naiv(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.dtype = dtype

        self.real_BatchNorm = nn.BatchNorm2d(self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.track_running_stats)
        self.imag_BatchNorm = nn.BatchNorm2d(self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.track_running_stats)

    def forward(self, input):
        return self.real_BatchNorm(input.real) + 1j * self.imag_BatchNorm(input.imag)  # torch.complex(self.real_BatchNorm(input.real), self.imag_BatchNorm(input.imag)) not used because of memory requirements


class Complex_BatchNorm1d_naiv(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.dtype = dtype

        self.real_BatchNorm = nn.BatchNorm1d(self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.track_running_stats, device=self.device, dtype=self.dtype)
        self.imag_BatchNorm = nn.BatchNorm1d(self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.track_running_stats, device=self.device, dtype=self.dtype)

    def forward(self, input):
        return self.real_BatchNorm(input.real) + 1j * self.imag_BatchNorm(input.imag)  # torch.complex(self.real_BatchNorm(input.real), self.imag_BatchNorm(input.imag)) not used because of memory requirements


class Complex_LayerNorm_naiv(nn.Module):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # self.device = device
        # self.dtype = dtype

        self.real_LayerNorm = nn.LayerNorm(self.normalized_shape, eps=self.eps, elementwise_affine=self.elementwise_affine)
        self.imag_LayerNorm = nn.LayerNorm(self.normalized_shape, eps=self.eps, elementwise_affine=self.elementwise_affine)

    def forward(self, input):
        return self.real_LayerNorm(input.real) + 1j * self.imag_LayerNorm(input.imag)  # torch.complex(self.real_LayerNorm(input.real), self.imag_LayerNorm(input.imag)) not used because of memory requirements


class Complex_Dropout(nn.Module):
    def __init__(self, p, inplace=False, size=None, device='cuda'):
        super().__init__()
        self.size = size
        self.device = device
        if self.size is not None:
            self.ones = torch.ones(size)
            if self.device is not None:
                self.ones = self.ones.to(self.device)
        self.real_dropout = nn.Dropout(p=p, inplace=inplace)

    def forward(self, input):
        if self.size is not None:
            return input * self.real_dropout(self.ones)
        else:
            if self.device is not None:
                return input * self.real_dropout(torch.ones(input.size()).to(self.device))
            return input * self.real_dropout(torch.ones(input.size()))


class Complex_LayerNorm(nn.Module):

    def __init__(self, embed_dim=None, eps=1e-05, elementwise_affine=True, device='cuda'):
        super().__init__()
        assert not(elementwise_affine and embed_dim is None), 'Give dimensions of learnable parameters or disable them'
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.embed_dim = embed_dim
            self.register_parameter(name='weights', param=torch.nn.Parameter(torch.empty([2, 2], dtype=torch.complex64)))
            self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(embed_dim, dtype=torch.complex64)))
            self.weights = torch.nn.Parameter(torch.eye(2))
            self.weights = torch.nn.Parameter((torch.Tensor([1, 1, 0]).repeat([embed_dim, 1])).unsqueeze(-1))
            self.bias = torch.nn.Parameter(torch.zeros([1, 1, embed_dim], dtype=torch.complex64))
        self.eps = eps

    def forward(self, input):

        ev = torch.unsqueeze(torch.mean(input, dim=-1), dim=-1)
        var_real = torch.unsqueeze(torch.unsqueeze(torch.var(input.real, dim=-1), dim=-1), dim=-1)
        var_imag = torch.unsqueeze(torch.unsqueeze(torch.var(input.imag, dim=-1), dim=-1), dim=-1)

        input = input - ev
        cov = torch.unsqueeze(torch.unsqueeze(torch.mean(input.real * input.imag, dim=-1), dim=-1), dim=-1)
        cov_m_0 = torch.cat((var_real, cov), dim=-1)
        cov_m_1 = torch.cat((cov, var_imag), dim=-1)
        cov_m = torch.unsqueeze(torch.cat((cov_m_0, cov_m_1), dim=-2), dim=-3)
        in_concat = torch.unsqueeze(torch.cat((torch.unsqueeze(input.real, dim=-1), torch.unsqueeze(input.imag, dim=-1)), dim=-1), dim=-1)

        cov_sqr = self.sqrt_2x2(cov_m)

        # out = self.inv_2x2(cov_sqr).matmul(in_concat)  # [..., 0]
        if self.elementwise_affine:
            real_var_weight = (self.weights[:, 0, :] ** 2).unsqueeze(-1).unsqueeze(0)
            imag_var_weight = (self.weights[:, 1, :] ** 2).unsqueeze(-1).unsqueeze(0)
            cov_weight = (torch.sigmoid(self.weights[:, 2, :].unsqueeze(-1).unsqueeze(0)) - 0.5) * 2 * torch.sqrt(real_var_weight * imag_var_weight)
            weights_mult = torch.cat([torch.cat([real_var_weight, cov_weight], dim=-1), torch.cat([cov_weight, imag_var_weight], dim=-1)], dim=-2).unsqueeze(0)
            mult_mat = self.sqrt_2x2(weights_mult).matmul(self.inv_2x2(cov_sqr))
            out = mult_mat.matmul(in_concat)  # makes new cov_m = self.weights
        else:
            out = self.inv_2x2(cov_sqr).matmul(in_concat)  # [..., 0]
        out = out[..., 0, 0] + 1j * out[..., 1, 0]  # torch.complex(out[..., 0], out[..., 1]) not used because of memory requirements
        if self.elementwise_affine:
            return out + self.bias
        return out

    def inv_2x2(self, input):
        a = torch.unsqueeze(torch.unsqueeze(input[..., 0, 0], dim=-1), dim=-1)
        b = torch.unsqueeze(torch.unsqueeze(input[..., 0, 1], dim=-1), dim=-1)
        c = torch.unsqueeze(torch.unsqueeze(input[..., 1, 0], dim=-1), dim=-1)
        d = torch.unsqueeze(torch.unsqueeze(input[..., 1, 1], dim=-1), dim=-1)
        divisor = a * d - b * c
        mat_1 = torch.cat((d, -b), dim=-2)
        mat_2 = torch.cat((-c, a), dim=-2)
        mat = torch.cat((mat_1, mat_2), dim=-1)
        return mat / divisor

    def sqrt_2x2(self, input):
        a = torch.unsqueeze(torch.unsqueeze(input[..., 0, 0], dim=-1), dim=-1)
        b = torch.unsqueeze(torch.unsqueeze(input[..., 0, 1], dim=-1), dim=-1)
        c = torch.unsqueeze(torch.unsqueeze(input[..., 1, 0], dim=-1), dim=-1)
        d = torch.unsqueeze(torch.unsqueeze(input[..., 1, 1], dim=-1), dim=-1)

        s = torch.sqrt(a * d - b * c)  # sqrt(det)
        t = torch.sqrt(a + d + 2 * s)  # sqrt(trace + 2 * sqrt(det))
        # maybe use 1/t * (M + sI) later, see Wikipedia

        return torch.cat((torch.cat((a + s, b), dim=-2), torch.cat((c, d + s), dim=-2)), dim=-1) / t


class Complex_BatchNorm1d(nn.Module):
    def __init__(self, num_channel=1, eps=1e-05, elementwise_affine=True, momentum=0.5, device='cuda'):
        super().__init__()
        assert not(elementwise_affine and num_channel is None), 'Give dimensions of learnable parameters or disable them'
        self.elementwise_affine = elementwise_affine
        self.register_buffer('running_mean', tensor=torch.zeros(num_channel, dtype=torch.complex64, requires_grad=False))
        self.register_buffer('running_cov', tensor=torch.eye(num_channel))
        self.momentum = momentum
        self.first_run = True

        if elementwise_affine:
            self.dim = num_channel
            self.register_parameter(name='weights', param=torch.nn.Parameter(torch.empty([num_channel, 1, 1, 3], dtype=torch.complex64)))
            self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(num_channel, dtype=torch.complex64)))
            self.weights = torch.nn.Parameter((torch.Tensor([1, 1, 0]).repeat([num_channel, 1])).unsqueeze(-2).unsqueeze(-1))
            self.bias = torch.nn.Parameter(torch.zeros([1, num_channel, 1], dtype=torch.complex64))
        # self.eps = eps

    def forward(self, input):
        if self.training:
            if input.shape[0] == 1:
                if self.first_run:
                    print('Batchnorm cant be calculated for Batchsize 1, proceds to ignore Batchnorm')
                    self.first_run = False
                return input

            ev = torch.unsqueeze(torch.unsqueeze(torch.mean(input, dim=[0, 2]), dim=0), dim=-1)
            if not self.first_run:
                self.running_mean = (ev * (1 - self.momentum) + self.running_mean * self.momentum).detach()
            else:
                self.running_mean = ev.detach()
            # self.running_mean = ev.type_as(self.running_mean).detach()

            var_real = torch.unsqueeze(torch.unsqueeze(torch.var(input.real, dim=[0, 2]), dim=-1), dim=-1) + 1e-9  # 1e-9 if variance 0
            var_imag = torch.unsqueeze(torch.unsqueeze(torch.var(input.imag, dim=[0, 2]), dim=-1), dim=-1) + 1e-9  # 1e-9 if variance 0

            input = input - ev
            cov = torch.unsqueeze(torch.unsqueeze(torch.mean(input.real * input.imag, dim=[0, 2]), dim=-1), dim=-1)
            cov_m_0 = torch.cat((var_real, cov), dim=-1)
            cov_m_1 = torch.cat((cov, var_imag), dim=-1)
            cov_m = torch.unsqueeze(torch.cat((cov_m_0, cov_m_1), dim=-2), dim=0)

            cov_m = self.inv_2x2(self.sqrt_2x2(cov_m))
            if self.first_run is False:
                self.running_cov = (cov_m * (1 - self.momentum) + self.running_cov * self.momentum).detach()  # note: running_cov is already sqrt and inv
            else:
                self.running_cov = cov_m.detach()
                self.first_run = False

        else:
            cov_m = self.running_cov
            ev = self.running_mean

            input = input - ev

        cov_m = torch.unsqueeze(cov_m, dim=-3)

        input = torch.unsqueeze(torch.view_as_real(input), dim=-1)

        if self.elementwise_affine:
            real_var_weight = (self.weights[:, :, 0, :] ** 2).unsqueeze(-1).unsqueeze(0)
            imag_var_weight = (self.weights[:, :, 1, :] ** 2).unsqueeze(-1).unsqueeze(0)
            cov_weight = (torch.sigmoid(self.weights[:, :, 2, :].unsqueeze(-1).unsqueeze(0)) - 0.5) * 2 * torch.sqrt(real_var_weight * imag_var_weight)
            weights_mult = torch.cat([torch.cat([real_var_weight, cov_weight], dim=-1), torch.cat([cov_weight, imag_var_weight], dim=-1)], dim=-2)
            mult_mat = cov_m.matmul(self.sqrt_2x2(weights_mult))
            out = mult_mat.matmul(input)
            # out = self.sqrt_2x2(weights_mult).matmul(out)  # makes new cov_m = self.weights
        else:
            out = cov_m.matmul(input)
        out = out[..., 0, 0] + 1j * out[..., 1, 0]  # torch.complex(out[..., 0, 0], out[..., 1, 0]) not used because of memory requirements
        if self.elementwise_affine:
            out = out + self.bias  # makes ev = bias
        return out

    def inv_2x2(self, input):
        a = torch.unsqueeze(torch.unsqueeze(input[..., 0, 0], dim=-1), dim=-1)
        b = torch.unsqueeze(torch.unsqueeze(input[..., 0, 1], dim=-1), dim=-1)
        c = torch.unsqueeze(torch.unsqueeze(input[..., 1, 0], dim=-1), dim=-1)
        d = torch.unsqueeze(torch.unsqueeze(input[..., 1, 1], dim=-1), dim=-1)
        divisor = a * d - b * c
        mat_1 = torch.cat((d, -b), dim=-2)
        mat_2 = torch.cat((-c, a), dim=-2)
        mat = torch.cat((mat_1, mat_2), dim=-1)
        return mat / divisor

    def sqrt_2x2(self, input):
        a = torch.unsqueeze(torch.unsqueeze(input[..., 0, 0], dim=-1), dim=-1)
        b = torch.unsqueeze(torch.unsqueeze(input[..., 0, 1], dim=-1), dim=-1)
        c = torch.unsqueeze(torch.unsqueeze(input[..., 1, 0], dim=-1), dim=-1)
        d = torch.unsqueeze(torch.unsqueeze(input[..., 1, 1], dim=-1), dim=-1)

        s = torch.sqrt(a * d - b * c)  # sqrt(det)
        t = torch.sqrt(a + d + 2 * s)  # sqrt(trace + 2 * sqrt(det))
        # maybe use 1/t * (M + sI) later, see Wikipedia

        return torch.cat((torch.cat((a + s, b), dim=-2), torch.cat((c, d + s), dim=-2)), dim=-1) / t


class Complex_BatchNorm2d(nn.Module):
    def __init__(self, num_channel=1, eps=1e-05, elementwise_affine=True, momentum=0.1, device='cuda'):
        super().__init__()
        assert not(elementwise_affine and num_channel is None), 'Give dimensions of learnable parameters or disable them'
        self.elementwise_affine = elementwise_affine
        self.register_buffer('running_mean', tensor=torch.zeros(num_channel, dtype=torch.complex64, requires_grad=False))
        self.register_buffer('running_cov', tensor=torch.Tensor([1, 1, 0]).repeat([num_channel, 1]))
        self.momentum = momentum
        self.first_run = True
        self.dim = num_channel

        if elementwise_affine:
            self.register_parameter(name='weights', param=torch.nn.Parameter(torch.empty([num_channel, 1, 1, 3], dtype=torch.float32)))
            self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(num_channel, dtype=torch.complex64)))
            self.weights = torch.nn.Parameter((torch.Tensor([1, 1, 0]).repeat([num_channel, 1])).unsqueeze(-2).unsqueeze(-2))
            self.bias = torch.nn.Parameter(torch.zeros([1, num_channel, 1, 1], dtype=torch.complex64))

    def forward(self, input):
        self.training = True
        if self.training:

            ev = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.mean(input, dim=[0, 2, 3]), dim=0), dim=-1), dim=-1)
            with torch.no_grad():
                if not self.first_run:
                    self.running_mean = (ev * (1 - self.momentum) + self.running_mean * self.momentum)
                else:
                    self.running_mean = ev

            cov_m = torch.zeros([3, self.dim, 1, 1], device='cuda')
            cov_m[0] = torch.unsqueeze(torch.unsqueeze(torch.var(input.real, dim=[0, 2, 3]), dim=-1), dim=-1) + 1e-9  # 1e-9 if variance 0
            cov_m[1] = torch.unsqueeze(torch.unsqueeze(torch.var(input.imag, dim=[0, 2, 3]), dim=-1), dim=-1) + 1e-9  # 1e-9 if variance 0

            input = input - ev
            cov_m[2] = torch.unsqueeze(torch.unsqueeze(torch.mean(input.real * input.imag, dim=[0, 2, 3]), dim=-1), dim=-1)

            cov_m = self.inv_sqrt_2x2(cov_m)  # self.inv_2x2(self.sqrt_2x2(cov_m))
            with torch.no_grad():
                if not self.first_run:
                    self.running_cov = (cov_m * (1 - self.momentum) + self.running_cov * self.momentum)  # note: running_cov is already sqrt and inv
                else:
                    self.running_cov = cov_m
                    self.first_run = False

        else:
            cov_m = self.running_cov
            ev = self.running_mean

            input = input - ev

        input = self.mult_2x2(input, cov_m)  # decorrelate input

        if self.elementwise_affine:
            real_var_weight = (self.weights[:, :, :, 0] ** 2).unsqueeze(0)
            imag_var_weight = (self.weights[:, :, :, 1] ** 2).unsqueeze(0)
            cov_weight = (torch.sigmoid(self.weights[:, :, :, 2].unsqueeze(0)) - 0.5) * 2 * torch.sqrt(real_var_weight * imag_var_weight)
            mult_weights = self.sqrt_2x2(torch.cat([real_var_weight, imag_var_weight, cov_weight], dim=0))

            input = self.mult_2x2(input, mult_weights)

        if self.elementwise_affine:
            return input + self.bias  # makes ev = bias
        return input

    def mult_2x2(self, input, mult):
        mult = torch.unsqueeze(mult, dim=0)
        return mult[:, 0] * input.real + mult[:, 2] * input.imag + 1j * (mult[:, 2] * input.real + mult[:, 1] * input.imag)

    def inv_sqrt_2x2(self, input):
        input = torch.unsqueeze(input, dim=0)
        s = torch.sqrt(input[:, 0] * input[:, 1] - input[:, 2] ** 2)
        t = torch.sqrt(input[:, 0] + input[:, 1] + 2 * s)
        return 1 / (t * s) * torch.cat([input[:, 1] + s, input[:, 0] + s, -input[:, 2]], dim=0)

    def sqrt_2x2(self, input):
        input = torch.unsqueeze(input, dim=0)
        s = torch.sqrt(input[:, 0] * input[:, 1] - input[:, 2] ** 2)
        t = torch.sqrt(input[:, 0] + input[:, 1] + 2 * s)
        return 1 / t * torch.cat([input[:, 0] + s, input[:, 1] + s, input[:, 2]], dim=0)


class Complex_MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.return_indices = return_indices
        self.real_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=True, ceil_mode=ceil_mode)

    def forward(self, input):
        pooled, indices = self.real_pool(torch.abs(input))
        if self.return_indices:
            return torch.reshape(input.flatten()[indices], pooled.shape), indices
        return torch.reshape(input.flatten()[indices], pooled.shape)
