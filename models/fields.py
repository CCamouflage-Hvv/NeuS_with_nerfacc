import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.8,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()
        """
        sdf_network {
        d_out = 257
        d_in = 3
        d_hidden = 256
        n_layers = 8
        skip_in = [4]
        multires = 6
        bias = 0.5
        scale = 1.0
        geometric_init = True
        weight_norm = True
        """
        self.__hidden__ = torch.nn.Linear(3, 1, bias=False)
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]#[3, 256, 256, 256, 256, 256, 256, 256, 256, 257]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001) #用正态分布初始化网络的权重
                        torch.nn.init.constant_(lin.bias, -bias)#用常数初始化网络的偏置
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        #print(f"inputs.grad_fn_begin:{inputs.grad_fn}")
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        #print(f"inputs.grad_fn_embed:{inputs.grad_fn}")
        x = inputs
        #print(f"x.grad_fn:{x.grad_fn}")
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
                #print(f"x.grad_fn_skip_in:{x.grad_fn}")
            x = lin(x)
            #print(f"x.grad_fn_lin:{x.grad_fn}")
            if l < self.num_layers - 2:
                x = self.activation(x)
                #print(f"x.grad_fn _activation:{x.grad_fn}")
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1) #x[:, :1] / self.scale是sdf值，x[:, 1:]是appearance。这个self.scale用于等比例缩放sdf这个场。

    def sdf(self, x):
        #print(f"x.grad_fn_sdf:{x.grad_fn}")
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)
    
        
    def gradient(self, x): #计算梯度,即sdf值对空间位置xyz的偏导数来求得的梯度,不需要考虑光线信息.
        #x.requires_grad_(True)
        x = x + 0 * self.__hidden__(x)
        y = self.forward(x)[:,:1]
        # print(x.grad_fn)  # None  x的积分方法
        # print(x.requires_grad)
        # print(y.grad_fn)  # None  x的积分方法
        # print(y.requires_grad)
        # print(f"pts x.shape:{x.shape}")
        # print(f"sdf y.shape:{y.shape}")
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
"""
renderingNetwork:
    
"""
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()
        """
        rendering_network {
        d_feature = 256
        mode = idr
        d_in = 9
        d_out = 3
        d_hidden = 256
        n_layers = 4
        weight_norm = True
        multires_view = 4
        squeeze_out = True
        """
        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]#[265, 256, 256, 256, 256, 3]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x #x是？


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
"""
NeRF field = 
    input: pos, viewdir
    output: rgb, sigma
    network: 
"""
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4], #跳过层
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D #deep = number of layers
        self.W = W #width of net
        self.d_in = d_in # pos input dimension
        self.d_in_view = d_in_view # view input dimension
        self.input_ch = 3 # pos channel
        self.input_ch_view = 3 # view input 
        self.embed_fn = None #pos embed fn
        self.embed_fn_view = None # view embed fn

        if multires > 0: # if use pos embed
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0: # if use view embed
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs # what is this?

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)]) # // 取整除 - 返回商的整数部分（向下取整）

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1) # skip_layer is used to concat the input and output of the previous layer

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb # here alpha is the density? rgb is the color.
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    """
    variance_network {
        init_val = 0.3
    }
    """
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
