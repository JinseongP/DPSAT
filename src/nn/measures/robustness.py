import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils


from .. import RobModel

class RobustnessMeasures:
    def __init__(self, rmodel):
        self.rmodel = rmodel.eval()
        
    def get_all_measures(self, clean_loader, adv_loader, eps):
        measures = OrderedDict()
        
        measures['CE_CLEAN'] = self.get_ce_loss(clean_loader, eps)
        measures['CE_ADV'] = self.get_ce_loss(adv_loader, eps)
        measures['KL_DIV'] = self.estimate_local_lip_v2(clean_loader, adv_loader, eps)
        
        measures['LOCAL_LIP'] = self.get_local_lip_v2(clean_loader, eps)
        measures['BOUNDARY_THICKNESS'] = self.get_boundary_thickness(clean_loader, eps)
    
    @torch.no_grad()
    def get_ce_loss(self, loader):
        device = next(self.rmodel.parameters()).device
        costs = []
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = self.rmodel(images)

            cost = nn.CrossEntropyLoss(reduction='none')(logits, labels)
            costs.append(cost)
        return torch.cat(costs)
    
    @torch.no_grad()
    def get_kl_loss(self, clean_loader, adv_loader):
        device = next(self.rmodel.parameters()).device
        costs = []
        for (images, labels), (adv_images, adv_labels) in zip(clean_loader, adv_loader):
            if labels != adv_labels:
                raise ValueError("Loaders are not matched.")
            images = images.to(device)
            adv_images = adv_images.to(device)
            labels = labels.to(device)
            
            logits_clean = self.rmodel(images)
            logits_adv = self.rmodel(adv_images)
            probs_clean = F.softmax(logits_clean, dim=1)
            log_probs_adv = F.log_softmax(logits_adv, dim=1)

            cost = nn.KLDivLoss(reduction='none')(log_probs_adv, probs_clean).sum(dim=1)
            costs.append(cost)
        return torch.cat(costs)
    
    # Modified from https://github.com/yangarbiter/robust-local-lipschitz/blob/master/lolip/utils.py
    def get_local_lip_v2(self, loader, eps=0.01, alpha=None, perturb_steps=10,
                              top_norm=1, btm_norm=float('inf'), device="cuda"):
        model = self.rmodel
        if alpha is None:
            step_size = eps / 3 
            
        def local_lip(model, x, xp, top_norm, btm_norm, reduction='mean'):
            model.eval()
            down = torch.flatten(x - xp, start_dim=1)
            if top_norm == "kl":
                criterion_kl = nn.KLDivLoss(reduction='none')
                top = criterion_kl(F.log_softmax(model(xp), dim=1),
                                   F.softmax(model(x), dim=1))
                ret = torch.sum(top, dim=1) / torch.norm(down + 1e-6, dim=1, p=btm_norm)
            else:
                top = torch.flatten(model(x), start_dim=1) - torch.flatten(model(xp), start_dim=1)
                ret = torch.norm(top, dim=1, p=top_norm) / torch.norm(down + 1e-6, dim=1, p=btm_norm)

            if reduction == 'mean':
                return torch.mean(ret)
            elif reduction == 'sum':
                return torch.sum(ret)
            else:
                raise ValueError(f"Not supported reduction: {reduction}")
                
        model.eval()

        total = 0.
        total_loss = 0.
        ret = []
        for x in loader:
            x = x[0]
            x = x.to(device)
            # generate adversarial example
            if btm_norm in [1, 2, np.inf]:
                x_adv = x + 0.001 * torch.randn(x.shape).to(device)

                # Setup optimizers
                optimizer = optim.SGD([x_adv], lr=step_size)

                for _ in range(perturb_steps):
                    x_adv.requires_grad_(True)
                    optimizer.zero_grad()
                    with torch.enable_grad():
                        loss = (-1) * local_lip(model, x, x_adv, top_norm, btm_norm)
                    loss.backward()
                    # renorming gradient
                    eta = step_size * x_adv.grad.data.sign().detach()
                    x_adv = x_adv.data.detach() + eta.detach()
                    eta = torch.clamp(x_adv.data - x.data, -eps, eps)
                    x_adv = x.data.detach() + eta.detach()
                    x_adv = torch.clamp(x_adv, 0, 1.0)
            else:
                raise ValueError(f"Unsupported norm {btm_norm}")

            total += x.shape[0]
            total_loss += local_lip(model, x, x_adv, top_norm, btm_norm, reduction='sum').item()
            print(total, end="\r")
    #         ret.append(x_adv.detach().cpu().numpy().transpose(0, 2, 3, 1))
            ret.append(x_adv.detach().cpu())
        ret_v = total_loss / total
        return ret_v, ret #np.concatenate(ret, axis=0)
    