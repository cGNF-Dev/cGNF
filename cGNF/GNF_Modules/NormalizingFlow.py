import torch
import torch.nn as nn
from .Conditionners import Conditioner, DAGConditioner
from .Normalizers import Normalizer
from torch.distributions.multivariate_normal import MultivariateNormal


class NormalizingFlow(nn.Module):
    def __init__(self):
        super(NormalizingFlow, self).__init__()

    '''
    Should return the x transformed and the log determinant of the Jacobian of the transformation
    '''
    def forward(self, x, context=None):
        pass

    '''
    Should return a term relative to the loss.
    '''
    def constraintsLoss(self):
        pass

    '''
    Should return the dagness of the associated graph.
    '''
    def DAGness(self):
        pass

    '''
    Step in the optimization procedure;
    '''
    def step(self, epoch_number, loss_avg):
        pass

    '''
    Return a list containing the conditioners.
    '''
    def getConditioners(self):
        pass

    '''
    Return True if the architecture is invertible.
    '''
    def isInvertible(self):
        pass

    '''
        Return a list containing the normalizers.
        '''

    def getNormalizers(self):
        pass

    '''
    Return the x that would generate z: [B, d] tensor.
    '''
    def invert(self, z, context=None):
        pass


class NormalizingFlowStep(NormalizingFlow):
    def __init__(self, conditioner: Conditioner, normalizer: Normalizer):
        super(NormalizingFlowStep, self).__init__()
        self.conditioner = conditioner
        self.normalizer = normalizer
        self.cat_dims = normalizer.cat_dims
        self.mu = normalizer.mu
        self.sigma = normalizer.sigma
        self.norm_jac = 0.0
        if normalizer.mu is not None or normalizer.sigma is not None:
            self.mu = nn.Parameter(normalizer.mu.data.clone(), requires_grad=False)
            self.sigma = nn.Parameter(normalizer.sigma.data.clone(), requires_grad=False)
            
            
            # normalization volume correction log-jacobian
            with torch.no_grad():
                self.norm_jac = -torch.log(self.sigma.data).sum().detach()

        if self.cat_dims is not None:
            self.U_noise = MultivariateNormal(torch.zeros(len(self.cat_dims)), torch.eye(len(self.cat_dims))/(6*4))
            
            
            
    def forward(self, x_unnorm, context=None):
        with torch.no_grad():
            x = (x_unnorm-self.mu.data)/self.sigma.data

#         print(f'f_x_mu=\n{self.mu.data}')
#         print(f'f_x_sigma=\n{self.sigma.data}')
#         print(f'f_x_unnorm=\n{x_unnorm[:4]}')
#         print(f'f_x=\n{x[:4]}')
        h = self.conditioner(x, context)
#         print(f'f_h=\n{h.sum()}')
    
#         print(f'f_x_mu=\n{self.mu.data}')
#         print(f'f_x_sigma=\n{self.sigma.data}')
#         print(f'f_x_unnorm=\n{x_unnorm[:4]}')
#         print(f'f_x=\n{x[:4]}')

        if self.training:
            if self.cat_dims is not None:
    #             keys = self.cat_dims.keys()
                keys = list(self.cat_dims)#.keys()
    #             print(keys)
                with torch.no_grad():
                    u_noise = torch.zeros(x.shape).to(x.device)
            #                     u_noise[:,keys] = torch.rand(torch.Size([x.shape[0],len(keys)])).to(x.device)#.float()
        #                 u_noise[:,keys] = torch.clamp(torch.randn(torch.Size([x.shape[0],len(keys)])).to(x.device)/6, max=1/2, min=-1/2)#.float()
#                     u_noise[:,keys] = self.U_noise.sample(torch.Size([x.shape[0]])).to(x.device)
#                     u_noise[:,keys] -= self.mu.data[keys]
#                     u_noise[:,keys] /= self.sigma.data[keys]
#                     u_noise[:,keys] = torch.normal(0.0, (self.sigma.data[keys].reciprocal()/6/4).expand(torch.Size([x.shape[0], len(keys)]))).to(x.device)
                    u_noise[:,keys] = torch.normal(0.0, (self.sigma.data[keys].reciprocal()/6).expand(torch.Size([x.shape[0], len(keys)]))).to(x.device)
#                     u_noise[:,keys] = torch.normal(0.0, (self.sigma.data[keys]/(6)).expand(torch.Size([x.shape[0], len(keys)]))).to(x.device)
#                     u_noise[:,keys] = torch.normal(u_noise[:,keys],1/(6*4)).float().to(x.device)
#                     u_noise[:,keys] = torch.normal(u_noise[:,keys],1/(6)).float().to(x.device)
#                 else:
#                     u_noise[:,keys] = torch.ones(torch.Size([x.shape[0],len(keys)])).to(x.device)*0.0#.float()
                x = x + u_noise 

        z, jac = self.normalizer(x, h, context)
#         print(f'f_z=\n{z[:4]}')
        
        
        return z, torch.log(jac).sum(1)+self.norm_jac

    def constraintsLoss(self):
        if type(self.conditioner) is DAGConditioner:
            return self.conditioner.loss()
        return 0.

    def DAGness(self):
        if type(self.conditioner) is DAGConditioner:
            return [self.conditioner.get_power_trace()]
        return [0.]

    def step(self, epoch_number, loss_avg):
        if type(self.conditioner) is DAGConditioner:
            self.conditioner.step(epoch_number, loss_avg)

    def getConditioners(self):
        return [self.conditioner]

    def getNormalizers(self):
        return [self.normalizer]

    def isInvertible(self):
        for conditioner in self.getConditioners():
            if not conditioner.is_invertible:
                return False
        return True

    def invert(self, z, context=None, do_idx=None, do_val=None):
#         print(f'i_z=\n{z[:4]}')
        with torch.no_grad():
            x = x_prev = torch.zeros_like(z)
#             x = (x_unnorm-self.mu.data)/self.sigma.data
            if self.cat_dims is not None:
                cat_dims = list(self.cat_dims.keys())
                n_cats = list(self.cat_dims.values())
                    
    #         i=0
    #         while torch.norm(x - x_prev) > 5e-4:
            for i in range(self.conditioner.depth() + 1):
#                 print(i, "/", self.conditioner.depth() + 1)
                if do_idx is not None and do_val is not None and len(do_idx) == do_val.shape[1]:
#                     x_unnorm[:, do_idx] = do_val
                    if self.mu is not None or self.sigma is not None:
                        x[:, do_idx] = (do_val-self.mu.data[do_idx])/self.sigma.data[do_idx]

    #             with torch.no_grad():
#                     x = (x_unnorm-self.mu.data)/self.sigma.data

    #             print(f'i_x_mu=\n{self.mu.data}')
    #             print(f'i_x_sigma=\n{self.sigma.data}')
    #             print(f'i_x_unnorm_{i}=\n{x_unnorm[:4]}')
    #             print(f'i_x_{i}=\n{x[:4]}')

    #                 if i in do_idx:
    #                     continue
                h = self.conditioner(x, context)
    #             print(f'i_h_{i}=\n{h.sum()}')
                x_prev = x
                x = self.normalizer.inverse_transform(z, h, context)
    #             if do_idx is not None and do_val is not None and len(do_idx) == do_val.shape[1]:
    #                 x[:, do_idx] = do_val
    #                 x_unnorm = (x*self.sigma.data)+self.mu.data

                if self.mu is not None or self.sigma is not None:
#                     with torch.no_grad():
                    if self.cat_dims is not None:
#                         cat_dims = list(self.cat_dims.keys())
#                         n_cats = list(self.cat_dims.values())
                        for cat_dim, n_cat in self.cat_dims.items():
                            x[:, cat_dim]=torch.abs(torch.clamp(torch.round((x[:, cat_dim]*self.sigma.data[cat_dim])+self.mu.data[cat_dim]), min=0.0, max=n_cat-1.0))
#                             x[:, cat_dim]=torch.abs(torch.round(torch.clamp((x[:, cat_dim]*self.sigma.data[cat_dim])+self.mu.data[cat_dim], min=0.0, max=n_cat-1.0)))
            #                 x[:, self.cat_dims]=torch.floor(x[:, self.cat_dims])
            #                 x[:, self.cat_dims]=(x[:, self.cat_dims] >= 1.0).float()
                        x[:, cat_dims] = (x[:, cat_dims]-self.mu.data[cat_dims])/self.sigma.data[cat_dims]
                
                if torch.norm(x - x_prev) == 0.0:
    #                 if self.mu is not None or self.sigma is not None:
    #                     with torch.no_grad():
    #                         x_unnorm = (x*self.sigma.data)+self.mu.data
                    break
    #             i+=1

    #         print(f'i_x_mu=\n{self.mu.data}')
    #         print(f'i_x_sigma=\n{self.sigma.data}')
    #         print(f'i_x_unnorm_final=\n{x_unnorm[:4]}')
    #         print(f'i_x_final=\n{x[:4]}')
#             x[:, cat_dims]=(x[:, cat_dims]*self.sigma.data[cat_dims])+self.mu.data[cat_dims]
            x = (x*self.sigma.data)+self.mu.data
            if self.cat_dims is not None:
                for cat_dim, n_cat in self.cat_dims.items():
                    x[:, cat_dim]=torch.abs(torch.clamp(torch.round(x[:, cat_dim]), min=0.0, max=n_cat-1.0))
            if do_idx is not None and do_val is not None and len(do_idx) == do_val.shape[1]:
                x[:, do_idx] = do_val

#                 x[:, cat_dim]=torch.abs(torch.round(torch.clamp((x[:, cat_dim]*self.sigma.data[cat_dim])+self.mu.data[cat_dim], min=0.0, max=n_cat-1.0)))
            
#             if do_idx is not None and do_val is not None and len(do_idx) == do_val.shape[1]:
#                 x[:, do_idx] = do_val
        return x

class FCNormalizingFlow(NormalizingFlow):
    def __init__(self, steps, z_log_density):
        super(FCNormalizingFlow, self).__init__()
        self.steps = nn.ModuleList()
        self.z_log_density = z_log_density
        for step in steps:
            self.steps.append(step)
        self.mu = step.mu
        self.sigma = step.sigma
        self.cat_dims = step.cat_dims

    def forward(self, x, context=None):
        jac_tot = 0.
        inv_idx = torch.arange(x.shape[1] - 1, -1, -1).long()
        for step in self.steps:
            step.normalizer.mu.data = self.mu
            step.normalizer.sigma.data = self.sigma
            z, jac = step(x, context)
            x = z[:, inv_idx]
            jac_tot += jac

        return z, jac_tot

    def constraintsLoss(self):
        loss = 0.
        for step in self.steps:
                loss += step.constraintsLoss()
        return loss

    def DAGness(self):
        dagness = []
        for step in self.steps:
            dagness += step.DAGness()
        return dagness

    def step(self, epoch_number, loss_avg):
        for step in self.steps:
            step.step(epoch_number, loss_avg)

    def loss(self, z, jac):
        log_p_x = jac + self.z_log_density(z)
        return self.constraintsLoss() - log_p_x.mean()

    def getNormalizers(self):
        normalizers = []
        for step in self.steps:
            normalizers += step.getNormalizers()
        return normalizers

    def getConditioners(self):
        conditioners = []
        for step in self.steps:
            conditioners += step.getConditioners()
        return conditioners

    def isInvertible(self):
        for conditioner in self.getConditioners():
            if not conditioner.is_invertible:
                return False
        return True

    def invert(self, z, context=None, do_idx=None, do_val=None):
        with torch.no_grad():
            for step in range(len(self.steps)):
                z = self.steps[-step].invert(z, context, do_idx=do_idx, do_val=do_val)
            return z


class CNNormalizingFlow(FCNormalizingFlow):
    def __init__(self, steps, z_log_density, dropping_factors):
        super(CNNormalizingFlow, self).__init__(steps, z_log_density)
        self.dropping_factors = dropping_factors

    def forward(self, x, context=None):
        b_size = x.shape[0]
        jac_tot = 0.
        z_all = []
        for step, drop_factors in zip(self.steps, self.dropping_factors):
            z, jac = step(x, context)
            d_c, d_h, d_w = drop_factors
            C, H, W = step.img_sizes
            c, h, w = int(C/d_c), int(H/d_h), int(W/d_w)
            z_reshaped = z.view(-1, C, H, W).unfold(1, d_c, d_c).unfold(2, d_h, d_h) \
                    .unfold(3, d_w, d_w).contiguous().view(b_size, c, h, w, -1)
            z_all += [z_reshaped[:, :, :, :, 1:].contiguous().view(b_size, -1)]
            x = z.view(-1, C, H, W).unfold(1, d_c, d_c).unfold(2, d_h, d_h) \
                    .unfold(3, d_w, d_w).contiguous().view(b_size, c, h, w, -1)[:, :, :, :, 0] \
                .contiguous().view(b_size, -1)
            jac_tot += jac
        z_all += [x]
        z = torch.cat(z_all, 1)
        return z, jac_tot

    def invert(self, z, context=None):
        b_size = z.shape[0]
        z_all = []
        i = 0
        for step, drop_factors in zip(self.steps, self.dropping_factors):
            d_c, d_h, d_w = drop_factors
            C, H, W = step.img_sizes
            c, h, w = int(C / d_c), int(H / d_h), int(W / d_w)
            nb_z = C*H*W - c*h*w if C*H*W != c*h*w else c*h*w
            z_all += [z[:, i:i+nb_z]]
            i += nb_z

        x = 0.
        for i in range(1, len(self.steps) + 1):
            step = self.steps[-i]
            drop_factors = self.dropping_factors[-i]
            d_c, d_h, d_w = drop_factors
            C, H, W = step.img_sizes
            c, h, w = int(C / d_c), int(H / d_h), int(W / d_w)
            z = z_all[-i]
            if c*h*w != C*H*W:
                z = z.view(b_size, c, h, w, -1)
                x = x.view(b_size, c, h, w, 1)
                z = torch.cat((x, z), 4)
                z = z.view(b_size, c, h, w, d_c, d_h, d_w)
                z = z.permute(0, 1, 2, 3, 6, 4, 5).contiguous().view(b_size, c, h, W, d_c, d_h)
                z = z.permute(0, 1, 2, 5, 3, 4).contiguous().view(b_size, c, H, W, d_c)
                z = z.permute(0, 1, 4, 2, 3).contiguous().view(b_size, C, H, W)
            x = step.invert(z.view(b_size, -1), context)
        return x

