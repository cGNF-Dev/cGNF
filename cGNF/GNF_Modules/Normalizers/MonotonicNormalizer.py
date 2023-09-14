import torch
from UMNN import NeuralIntegral, ParallelNeuralIntegral
from .Normalizer import Normalizer
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class ELUPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1.05


class IntegrandNet(nn.Module):
    def __init__(self, hidden, cond_in):
        super(IntegrandNet, self).__init__()
        l1 = [1 + cond_in] + hidden
        l2 = hidden + [1]
        layers = []
        for h1, h2 in zip(l1, l2):
#             layers += [nn.Linear(h1, h2), ELUPlus()]
            layers += [nn.Linear(h1, h2), nn.ReLU()]
        layers.pop()
        layers.append(ELUPlus())
        self.net = nn.Sequential(*layers)

    def forward(self, x, h):
        nb_batch, in_d = x.shape
        x = torch.cat((x, h), 1)
        x_he = x.view(nb_batch, -1, in_d).transpose(1, 2).contiguous().view(nb_batch * in_d, -1)
        y = self.net(x_he).view(nb_batch, -1)
        return y


class MonotonicNormalizer(Normalizer):
    def __init__(self, integrand_net, cond_size, nb_steps=20, solver="CC", mu=None, sigma=None, cat_dims=None):
        super(MonotonicNormalizer, self).__init__()
        if type(integrand_net) is list:
            self.integrand_net = IntegrandNet(integrand_net, cond_size)
        else:
            self.integrand_net = integrand_net
        self.solver = solver
        self.nb_steps = nb_steps
        self.cat_dims = cat_dims
        self.mu = mu
        self.sigma = sigma
        if mu is not None or sigma is not None:
            self.mu = nn.Parameter(mu, requires_grad=False)
            self.sigma = nn.Parameter(sigma, requires_grad=False)
        
        if self.cat_dims is not None:
            self.U_noise = MultivariateNormal(torch.zeros(len(self.cat_dims)), torch.eye(len(self.cat_dims))/(6*4))



    def forward(self, x, h, context=None):
        
        x0 = torch.zeros(x.shape).to(x.device)
        if self.cat_dims is not None:
#             keys = self.cat_dims.keys()
            keys = list(self.cat_dims)#.keys()
#             print(keys)
            with torch.no_grad():
                u_noise = torch.zeros(x.shape).to(x.device)
#                 if self.training:
# #                     u_noise[:,keys] = torch.rand(torch.Size([x.shape[0],len(keys)])).to(x.device)#.float()
#                     u_noise[:,keys] = torch.clamp(torch.randn(torch.Size([x.shape[0],len(keys)])).to(x.device)/6, max=1/2, min=-1/2)#.float()
#                     u_noise[:,keys] = torch.clamp(self.U_noise.sample(torch.Size([x.shape[0]])).to(x.device), max=1/2, min=-1/2)#.float()
    
#                 else:
#                     u_noise[:,keys] = torch.ones(torch.Size([x.shape[0],len(keys)])).to(x.device)*0.0#.float()
#             print(u_noise[:2,:])
            xT = x + u_noise
        else:
            xT = x
        
        z0 = h[:, :, 0]
        h = h.permute(0, 2, 1).contiguous().view(x.shape[0], -1)

        if self.solver == "CC":
            z = NeuralIntegral.apply(x0, xT, self.integrand_net, _flatten(self.integrand_net.parameters()),
                                     h, self.nb_steps) + z0
        elif self.solver == "CCParallel":
            z = ParallelNeuralIntegral.apply(x0, xT, self.integrand_net,
                                             _flatten(self.integrand_net.parameters()),
                                             h, self.nb_steps) + z0
        else:
            return None
        return z, self.integrand_net(x, h)


    def inverse_transform(self, z, h, context=None):
        x_max = torch.ones_like(z) * 20
        x_min = -torch.ones_like(z) * 20
        z_max, _ = self.forward(x_max, h, context)
        z_min, _ = self.forward(x_min, h, context)
        for i in range(30):
#         i=0
#         while (torch.abs(x_max-x_min)>1e-4).float().sum()!= 0.0:
#             if i>30:
#                 break
            x_middle = (x_max + x_min) / 2
            z_middle, _ = self.forward(x_middle, h, context)
            left = (z_middle > z).float()
            right = 1 - left
            x_max = left * x_middle + right * x_max
            x_min = right * x_middle + left * x_min
            z_max = left * z_middle + right * z_max
            z_min = right * z_middle + left * z_min
#             print(f'{i} : X_min = {x_min.cpu().numpy()}, X_max = {x_max.cpu().numpy()}, abs = {torch.abs(x_max-x_min).sum()}')
#             print(f'{i} : abs = {torch.abs(x_max-x_min).sum()}')

#             i+=1
        return (x_max + x_min) / 2
