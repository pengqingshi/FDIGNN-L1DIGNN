import torch
import torch.nn as nn
import torch.autograd as autograd

from torch_scatter import scatter_add

from src.solvers import *


class DEQFixedPoint(nn.Module):
    def __init__(self,
                 func,
                 fw_solver,
                 bw_solver,
                 max_iter,
                 threshold,
                 **kwargs):
        super(DEQFixedPoint, self).__init__()

        self.func = func
        self.fw_solver = fw_solver
        self.bw_solver = bw_solver
        self.max_iter = max_iter
        self.threshold = threshold
        self.kwargs = kwargs

    def forward(self, x, edge_index, edge_weight=None):
        with torch.no_grad():
            z, self.fw_res = self.fw_solver(lambda z: self.func(x, z, edge_index, edge_weight),
                                         torch.zeros_like(x), self.threshold, self.max_iter)
        z = self.func(x, z, edge_index, edge_weight)

        if self.training:
            z0 = z.clone().detach().requires_grad_()
            f0 = self.func(x, z0, edge_index, edge_weight)
            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()
                def ff(y):
                    dd = autograd.grad(f0, z0, y, retain_graph=True)
                    return dd[0] + grad
                g, self.backward_res = self.bw_solver(ff,
                                                torch.zeros_like(x), self.threshold, self.max_iter)
                return g
            
            self.hook = z.register_hook(backward_hook)
        return z


class ImplicitFunc_Neural(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mu,
                 epsilon=1e-6):
        super(ImplicitFunc_Neural, self).__init__()

        self.mu = mu
        self.epsilon = epsilon

        self.Theta_chi = nn.Linear(in_channels, out_channels, bias=False)
        self.Theta_phi = nn.Linear(out_channels, out_channels, bias=False)
        self.Theta_varphi = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, z, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        row, col = edge_index

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        degree = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)


        Phi_chi = torch.tanh(torch.norm(self.Theta_chi(x), dim=1))

        x_src = self.Theta_phi(self.Theta_chi(x[row,:])).unsqueeze(1)
        x_dst = self.Theta_phi(self.Theta_chi(x[col,:])).unsqueeze(1)
        Phi_phi = torch.tanh(torch.abs(torch.einsum("ijk,ijk->i", [x_src, x_dst])))

        Phi_varphi = torch.tanh(1 / (torch.norm(self.Theta_varphi(x[row,:] - x[col,:]), dim=1) + self.epsilon))

        Phi = (edge_weight / degree[row]) * Phi_phi * Phi_varphi / Phi_chi[row]

        energies = Phi.unsqueeze(1) * (z[row,:] - z[col,:])
        
        z_star = scatter_add(energies, row, dim=0, dim_size=num_nodes)
        z_star = x - 1/self.mu * z_star

        return z_star


class ImplicitFunc_Neural_Simplified(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mu,
                 epsilon=1e-6):
        super(ImplicitFunc_Neural_Simplified, self).__init__()

        self.mu = mu
        self.epsilon = epsilon

        self.Theta_chi = nn.Linear(in_channels, out_channels, bias=False)
        self.Theta_phi = nn.Linear(out_channels, out_channels, bias=False)
        self.Theta_varphi = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        row, col = edge_index

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        degree = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        Phi_chi = torch.tanh(torch.norm(self.Theta_chi(x), dim=1))

        x_src = self.Theta_phi(self.Theta_chi(x[row, :])).unsqueeze(1)
        x_dst = self.Theta_phi(self.Theta_chi(x[col, :])).unsqueeze(1)
        Phi_phi = torch.tanh(torch.abs(torch.einsum("ijk,ijk->i", [x_src, x_dst])))

        Phi_varphi = torch.tanh(
            1 / (torch.norm(self.Theta_varphi(x[row, :] - x[col, :]), dim=1) + self.epsilon))

        Phi = (edge_weight / degree[row]) * Phi_phi * Phi_varphi / Phi_chi[row]

        energies = Phi.unsqueeze(1) * (x[row, :] - x[col, :])

        x_star = scatter_add(energies, row, dim=0, dim_size=num_nodes)
        z = x - 1 / self.mu * x_star

        return z


class ImplicitFunc_Neural_Simplified_L1(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 rho,
                 mu_k_max,
                 mu_k,
                 K_iter,
                 epsilon=1e-6):
        super(ImplicitFunc_Neural_Simplified_L1, self).__init__()
        self.rho = rho
        self.mu_k_max = mu_k_max
        self.mu_k = mu_k
        self.K_iter = K_iter
        self.epsilon = epsilon

        self.Theta_chi = nn.Linear(in_channels, out_channels, bias=False)
        self.Theta_phi = nn.Linear(out_channels, out_channels, bias=False)
        self.Theta_varphi = nn.Linear(in_channels, out_channels, bias=False)


    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        row, col = edge_index

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        degree = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        Phi_chi = torch.tanh(torch.norm(self.Theta_chi(x), dim=1))

        x_src = self.Theta_phi(self.Theta_chi(x[row, :])).unsqueeze(1)
        x_dst = self.Theta_phi(self.Theta_chi(x[col, :])).unsqueeze(1)
        Phi_phi = torch.tanh(torch.abs(torch.einsum("ijk,ijk->i", [x_src, x_dst])))

        Phi_varphi = torch.tanh(
            1 / (torch.norm(self.Theta_varphi(x[row, :] - x[col, :]), dim=1) + self.epsilon))

        Delta = (edge_weight / degree[row]) * Phi_phi * Phi_varphi / Phi_chi[row]

        Delta = Delta.unsqueeze(1)

        z = ADMM(x, Delta, self.rho, self.mu_k_max, self.mu_k, self.K_iter, edge_index)

        return z
    