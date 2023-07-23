# Modified from https://github.com/nitarshan/robust-generalization-measures
from contextlib import contextmanager
from copy import deepcopy
from collections import OrderedDict
import math
import numpy as np

import torch
from .. import RobModel

class GeneralizationMeasures:
    def __init__(self, rmodel, init_info=None, remove_batchnorm=True):
        self.rmodel = rmodel.eval()
        
        self.rmodel_init = None
        if init_info is not None:
            # Get init model from path or dictionary.
            if isinstance(init_info, RobModel):
                self.rmodel_init = init_info.eval()
            # Get init model from path or dictionary.
            elif isinstance(init_info, str) or isinstance(init_info, dict) or isinstance(init_info, OrderedDict):
                self.rmodel_init = deepcopy(rmodel)
                self.rmodel_init.load_state_dict_auto(init_info)
                self.rmodel_init = self.rmodel_init.eval()
            else:
                raise ValueError('Invalid init_dict')
        
        # Remove batchnorm
        if remove_batchnorm:
            self.rmodel = _reparam(self.rmodel)
            if self.rmodel_init is not None:
                self.rmodel_init = _reparam(self.rmodel_init)
                
        # Extract weights only
        self._weights = _get_weights_only(self.rmodel)
        self._w_vec = _get_vec_params(self._weights)
        self.depth = len(self._weights)
        self.num_params = len(self._w_vec)

        # Extract init weights
        if self.rmodel_init is not None:
            self._dist_init_weights = [p-q for p,q in zip(self._weights, _get_weights_only(self.rmodel_init))]
            self._dist_w_vec = _get_vec_params(self._dist_init_weights)

    def get_all_measures(self, data_loader=None):
        measures = OrderedDict()

        if data_loader:
            m = _get_len_loader(data_loader)
            for data, target in data_loader:
                input_shape = data.shape[1:]
                break

        print("Vector Norm Measures")
        measures['L2'] = self.get_L2()
        measures['L2_DIST'] = self.get_L2_DIST() # init_model
        measures['PARAMS'] = self.num_params

        print("Measures on the output of the network")
        margins = self.get_margin(data_loader, m)
        
        measures['MARGIN_10TH_PERCENTILE'] = margins.kthvalue(m // 10)[0]
        measures['MARGIN_25TH_PERCENTILE'] = margins.kthvalue(m // 25)[0]
        measures['MARGIN_50TH_PERCENTILE'] = margins.kthvalue(m // 50)[0]
        measures['MARGIN_75TH_PERCENTILE'] = margins.kthvalue(m // 75)[0]
        measures['MARGIN_90TH_PERCENTILE'] = margins.kthvalue(m // 90)[0]
        
        measures['MARGIN_MAX'] = margins.max()
        measures['MARGIN_AVERAGE'] = margins.mean()
        measures['MARGIN_MIN'] = margins.min()
        
        margin = margins.kthvalue(m // 10)[0].abs() # 10th-percentile of margin
        measures['INVERSE_MARGIN'] = 1 / margin ** 2

        print("(Norm & Margin)-Based Measures")
        fro_norms = torch.cat([p.view(p.shape[0],-1).norm('fro').unsqueeze(0) ** 2 for p in self._weights])
        spec_norms = torch.cat([p.view(p.shape[0],-1).svd().S.max().unsqueeze(0) ** 2 for p in self._weights])
        dist_fro_norms = torch.cat([p.view(p.shape[0],-1).norm('fro').unsqueeze(0) ** 2 for p in self._dist_init_weights])
        dist_spec_norms = torch.cat([p.view(p.shape[0],-1).svd().S.max().unsqueeze(0) ** 2 for p in self._dist_init_weights])        

        print("-- Approximate Spectral Norm")
        measures['LOG_PROD_OF_SPEC'] = spec_norms.log().sum()
        measures['LOG_PROD_OF_SPEC_OVER_MARGIN'] = measures['LOG_PROD_OF_SPEC'] - 2 * margin.log()
        measures['LOG_SPEC_INIT_MAIN'] = measures['LOG_PROD_OF_SPEC_OVER_MARGIN'] + (dist_fro_norms / spec_norms).sum().log()
        measures['FRO_OVER_SPEC'] = (fro_norms / spec_norms).sum()
        measures['LOG_SPEC_ORIG_MAIN'] = measures['LOG_PROD_OF_SPEC_OVER_MARGIN'] + measures['FRO_OVER_SPEC'].log()
        measures['LOG_SUM_OF_SPEC_OVER_MARGIN'] = math.log(self.depth) + (1/self.depth) * (measures['LOG_PROD_OF_SPEC'] -  2 * margin.log())
        measures['LOG_SUM_OF_SPEC'] = math.log(self.depth) + (1/self.depth) * measures['LOG_PROD_OF_SPEC']

        print("-- Frobenius Norm")
        measures['LOG_PROD_OF_FRO'] = fro_norms.log().sum()
        measures['LOG_PROD_OF_FRO_OVER_MARGIN'] = measures['LOG_PROD_OF_FRO'] -  2 * margin.log()
        measures['LOG_SUM_OF_FRO_OVER_MARGIN'] = math.log(self.depth) + (1/self.depth) * (measures['LOG_PROD_OF_FRO'] -  2 * margin.log())
        measures['LOG_SUM_OF_FRO'] = math.log(self.depth) + (1/self.depth) * measures['LOG_PROD_OF_FRO']

        print("-- Distance to Initialization")
        measures['FRO_DIST'] = dist_fro_norms.sum()
        measures['DIST_SPEC_INIT'] = dist_spec_norms.sum()
        measures['PARAM_NORM'] = fro_norms.sum()

        print("Path-norm")
        measures['PATH_NORM'] = self.get_path_norm(input_shape)
        measures['PATH_NORM_OVER_MARGIN'] = measures['PATH_NORM'] / margin ** 2

        print("Flatness-based measures")
        measures['PACBAYES_SIGMA'] = self.get_pacbayes_sigma(data_loader, m)
        measures['PACBAYES_INIT'] = self._get_pacbayes_bound(self._dist_w_vec, measures['PACBAYES_SIGMA'], m)
        measures['PACBAYES_ORIG'] = self._get_pacbayes_bound(self._w_vec, measures['PACBAYES_SIGMA'], m)
        measures['PACBAYES_FLATNESS'] = torch.tensor(1 / measures['PACBAYES_SIGMA'] ** 2) 

        print("Magnitude-aware Perturbation Bounds")
        mag_eps = 1e-3
        omega = self.num_params
        measures['PACBAYES_MAG_SIGMA'] = self.get_pacbayes_sigma(data_loader, m, mag_eps)
        measures['PACBAYES_MAG_INIT'] = self._get_pacbayes_mag_bound(self._dist_w_vec, mag_eps, measures['PACBAYES_MAG_SIGMA'], omega, m)
        measures['PACBAYES_MAG_ORIG'] = self._get_pacbayes_mag_bound(self._w_vec, mag_eps, measures['PACBAYES_MAG_SIGMA'], omega, m)
        measures['PACBAYES_MAG_FLATNESS'] = torch.tensor(1 / measures['PACBAYES_MAG_SIGMA'] ** 2)

        return measures
    
#         # TODO: Adjust for dataset size
#         def adjust_measure(measure, value):
#             if measure.name.startswith('LOG_'):
#                 return 0.5 * (value - np.log(m))
#             else:
#                 return np.sqrt(value / m)
#         return {k: adjust_measure(k, v.item()) for k, v in measures.items()}

    # TODO: Augmentation까지 포함하는 게 맞나? 일단 포함.
    @torch.no_grad()
    def get_path_norm(self, input_shape):
        rmodel = deepcopy(self.rmodel)
        rmodel.eval()
        for param in rmodel.parameters():
            if param.requires_grad:
                param.data.pow_(2)
        x = torch.ones([1] + list(input_shape))
        x = rmodel(x)
        del rmodel
        return x.sum()

    @torch.no_grad()
    def get_L2(self):
        return self._w_vec.norm(p=2)

    @torch.no_grad()
    def get_L2_DIST(self):
        return self._dist_w_vec.norm(p=2)

    @torch.no_grad()
    def get_margin(self, data_loader, m=None):
        if m is None:
            m = _get_len_loader(data_loader)

        device = next(self.rmodel.parameters()).device
        margins = []
        for data, target in data_loader:
            logits = self.rmodel(data.to(device))
            correct_logit = logits[torch.arange(logits.shape[0]), target].clone()
            logits[torch.arange(logits.shape[0]), target] = float('-inf')
            max_other_logit = logits.data.max(1).values  # get the index of the max logits
            margin = correct_logit - max_other_logit
            margins.append(margin)
        return torch.cat(margins)

    @torch.no_grad()
    def get_pacbayes_sigma(self, data_loader, m=None, magnitude_eps=None,
                           search_depth=15, montecarlo_samples=10, accuracy_displacement=0.1, 
                           displacement_tolerance=1e-2, seed=0):
        lower, upper = 0, 2
        sigma = 1
        accuracy = self.rmodel.eval_accuracy(data_loader)
        if m is None:
            m = _get_len_loader(data_loader)

        BIG_NUMBER = 10348628753
        device = next(self.rmodel.parameters()).device
        rng = torch.Generator(device=device) if magnitude_eps is not None else torch.Generator()
        rng.manual_seed(BIG_NUMBER + seed)

        for _ in range(search_depth):
            sigma = (lower + upper) / 2
            accuracy_samples = []
            for _ in range(montecarlo_samples):
                with _perturbed_model(self.rmodel, sigma, rng, magnitude_eps) as p_model:
                    loss_estimate = 0
                    for data, target in data_loader:
                        logits = p_model(data.to(device))
                        pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
                        batch_correct = pred.cpu().eq(target.cpu().data.view_as(pred)).type(torch.FloatTensor).cpu()
                        loss_estimate += batch_correct.sum()
                        loss_estimate /= m
                        accuracy_samples.append(loss_estimate)
                    displacement = abs(np.mean(accuracy_samples) - accuracy)
            if abs(displacement - accuracy_displacement) < displacement_tolerance:
                break
            elif displacement > accuracy_displacement:
                # Too much perturbation
                upper = sigma
            else:
                # Not perturbed enough to reach target displacement
                lower = sigma
        return sigma

    @torch.no_grad()
    def _get_pacbayes_mag_bound(self, reference_vec, mag_eps, mag_sigma, omega, m):
        numerator = mag_eps ** 2 + (mag_sigma ** 2 + 1) * (reference_vec.norm(p=2)**2) / omega
        denominator = mag_eps ** 2 + mag_sigma ** 2 * self._dist_w_vec ** 2
        return 1/4 * (numerator / denominator).log().sum() + math.log(m / mag_sigma) + 10

    @torch.no_grad()
    def _get_pacbayes_bound(self, reference_vec, sigma, m):
        return (reference_vec.norm(p=2) ** 2) / (4 * sigma ** 2) + math.log(m / sigma) + 10

    
# Adapted from https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
@torch.no_grad()
def _reparam(model):
    def in_place_reparam(model, prev_layer=None):
        for child in model.children():
            prev_layer = in_place_reparam(child, prev_layer)
            if child._get_name() == 'Conv2d':
                prev_layer = child
            elif child._get_name() == 'BatchNorm2d':
                scale = child.weight / ((child.running_var + child.eps).sqrt())
                if prev_layer.bias:
                    prev_layer.bias.copy_( child.bias  + ( scale * (prev_layer.bias - child.running_mean) ) )
                perm = list(reversed(range(prev_layer.weight.dim())))
                prev_layer.weight.copy_((prev_layer.weight.permute(perm) * scale ).permute(perm))
                child.bias.fill_(0)
                child.weight.fill_(1)
                child.running_mean.fill_(0)
                child.running_var.fill_(1)
        return prev_layer
    model = deepcopy(model)
    in_place_reparam(model)
    return model


@contextmanager
def _perturbed_model(model, sigma, rng, magnitude_eps=None):
    device = next(model.parameters()).device
    if magnitude_eps is not None:
        noise = [torch.normal(0,sigma**2 * torch.abs(p) ** 2 + magnitude_eps ** 2, generator=rng) for p in model.parameters()]
    else:
        noise = [torch.normal(0,sigma**2,p.shape, generator=rng).to(device) for p in model.parameters()]
    model = deepcopy(model)
    try:
        [p.add_(n) for p,n in zip(model.parameters(), noise)]
        yield model
    finally:
        [p.sub_(n) for p,n in zip(model.parameters(), noise)]
        del model


def _get_len_loader(data_loader):
    m = 0
    for items in data_loader:
        m += len(items[0])
    return m


def _get_weights_only(model):
    blacklist = {'bias', 'bn'}
    return [p for name, p in model.named_parameters() if all(x not in name for x in blacklist)]


def _get_vec_params(weights):
    return torch.cat([p.view(-1) for p in weights], dim=0)