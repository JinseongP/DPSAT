import torch
from collections import defaultdict
from typing import Callable, List, Optional, Union        

from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _generate_noise


"""
Minimizers for DP
DPSAT: reuse the perturbed gradients of previous iteration.
DPSATMomentum: reuse the momentum of the previous gradients, i.e., reuse the direction w.r.t the previous weight parameters.
                Refer to Park, Jinseong, et al. "Fast sharpness-aware training for periodic time series classification and forecasting." 
                Applied Soft Computing (2023): 110467. (https://www.sciencedirect.com/science/article/pii/S1568494623004854).

Modified from https://github.com/SamsungLabs/ASAM    
"""

class DPSAT:
    def __init__(self, optimizer, model, rho=0.5):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        rho = self.rho
        grads = []
        self.grads_ascent = []

        for n, p in self.model.named_parameters():
            prev_grad = self.state[p].get("prev_grad")
            if prev_grad is None:
                prev_grad = torch.zeros_like(p)
                self.state[p]["prev_grad"] = prev_grad
            self.grads_ascent.append(self.state[p]["prev_grad"].flatten())
            grads.append(torch.norm(prev_grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2)
        if grad_norm != 0:
            grad_norm = grad_norm + 1.e-16 
        else:
            grad_norm.fill_(1)

        for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 continue
            eps = self.state[p].get("prev_grad")
            self.state[p]["eps"] = eps

            eps.mul_(rho / grad_norm)
            p.add_(eps)
             
        self.optimizer.zero_grad()        
        
    @torch.no_grad()
    def descent_step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.optimizer.pre_step():          
            self.grads_descent = []
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["eps"])
                prev_grad = p.grad.clone().detach()
                self.state[p]["prev_grad"] = prev_grad
                self.grads_descent.append(self.state[p]["prev_grad"].flatten())

            self.optimizer.original_optimizer.step()
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        
class DPSATMomentum:
    def __init__(self, optimizer, model, rho=0.5):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        rho = self.rho
        grads = []
        for n, p in self.model.named_parameters():
            prev_p = self.state[p].get("prev")
            if prev_p is None:
                prev_p = torch.clone(p).detach()
                self.state[p]["prev"] = prev_p
            grads.append(torch.norm(prev_p - p, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2)
        if grad_norm != 0:
            grad_norm = grad_norm + 1.e-16 
        else:
            grad_norm.fill_(1)

        for n, p in self.model.named_parameters():
            prev_p = self.state[p].get("prev")
            eps = prev_p - p
            self.state[p]["eps"] = eps

            eps.mul_(rho / grad_norm)
            p.add_(eps)
             
        self.optimizer.zero_grad()        
        
    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
            prev_p = torch.clone(p).detach()
            self.state[p]["prev"] = prev_p
        self.optimizer.step()
        self.optimizer.zero_grad()
 

"""
The below minimizers are not modified for DP versions.
"""

# class ASAM:

#     def __init__(self, optimizer, model, rho=0.5, eta=0.01):
#         self.optimizer = optimizer
#         self.model = model
#         self.rho = rho
#         self.eta = eta
#         self.state = defaultdict(dict)

#     @torch.no_grad()
#     def ascent_step(self):
#         wgrads = []
#         for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 continue
#             t_w = self.state[p].get("eps")
#             if t_w is None:
#                 t_w = torch.clone(p).detach()
#                 self.state[p]["eps"] = t_w
#             if 'weight' in n:
#                 t_w[...] = p[...]
#                 t_w.abs_().add_(self.eta)
#                 p.grad.mul_(t_w)
#             wgrads.append(torch.norm(p.grad, p=2))
#         wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
#         for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 continue
#             t_w = self.state[p].get("eps")
#             if 'weight' in n:
#                 p.grad.mul_(t_w)
#             eps = t_w
#             eps[...] = p.grad[...]
#             eps.mul_(self.rho / wgrad_norm)
#             p.add_(eps)
#         self.optimizer.zero_grad()

#     @torch.no_grad()
#     def descent_step(self):
#         for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 continue
#             p.sub_(self.state[p]["eps"])
#         self.optimizer.step()
#         self.optimizer.zero_grad()


# class SAM(ASAM):
#     @torch.no_grad()
#     def ascent_step(self):
#         grads = []
#         for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 continue
#             grads.append(torch.norm(p.grad, p=2))
#         grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
#         for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 continue
#             eps = self.state[p].get("eps")
#             if eps is None:
#                 eps = torch.clone(p).detach()
#                 self.state[p]["eps"] = eps
#             eps[...] = p.grad[...]
#             eps.mul_(self.rho / grad_norm)
#             p.add_(eps)
#         self.optimizer.zero_grad()

# class GSAM:
#     def __init__(self, optimizer, model, rho=0.5, alpha=0.1,
#                  decay=True, rho_min=0, lr_max=None, lr_min=0):
#         self.optimizer = optimizer
#         self.model = model
#         self.rho = rho
#         self.state = defaultdict(dict)
#         self.alpha = alpha
        
#         self.decay = decay
#         assert self.decay and (lr_max is not None)
#         self.rho_min = rho_min
#         self.lr_max = lr_max
#         self.lr_min = lr_min
        
#     @torch.no_grad()
#     def ascent_step(self):
#         grads = []
#         for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 continue
#             self.state["ascent"][n] = p.grad.clone().detach()
#             grads.append(torch.norm(p.grad, p=2))
#         grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
#         self.ascent_norm = grad_norm.clone().detach()
        
#         if self.decay:
#             lr = self.optimizer.param_groups[0]['lr']
#             rho = self.rho_min + (self.rho - self.rho_min)*(lr - self.lr_min)/(self.lr_max - self.lr_min)
#         else:
#             rho = self.rho
        
#         for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 continue
#             eps = self.state[p].get("eps")
#             if eps is None:
#                 eps = torch.clone(p).detach()
#                 self.state[p]["eps"] = eps
#             eps[...] = p.grad[...]
#             eps.mul_(rho / grad_norm)
#             p.add_(eps)
#         self.optimizer.zero_grad()
        
#     @torch.no_grad()
#     def descent_step(self):
#         self.state["descent"] = {}
#         grads = []
#         for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 continue
#             self.state["descent"][n] = p.grad.clone().detach()
#             grads.append(torch.norm(p.grad, p=2))
#         grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
#         self.descent_norm = grad_norm
        
#         descents = self.state["descent"]
#         ascents = self.state["ascent"]
        
#         inner_prod = self.inner_product(descents, ascents)
        
#         # get cosine
#         cosine = inner_prod / (self.ascent_norm * self.descent_norm + 1.e-16)
        
#         for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 continue
#             vertical = ascents[n] - cosine * self.ascent_norm * descents[n] / (self.descent_norm + 1.e-16)
#             p.grad.sub_(self.alpha*vertical)

#         self.optimizer.step()
#         self.optimizer.zero_grad()

#     def inner_product(self, u, v):
#         value = 0
#         for key in v.keys():
#             value += torch.dot(u[key].flatten(),v[key].flatten())
#         return value        