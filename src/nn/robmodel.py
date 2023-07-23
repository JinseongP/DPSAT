from collections import OrderedDict

import torch
import torch.nn as nn

from .modules.normalize import Normalize
from ..utils import get_accuracy


class RobModel(nn.Module):
    r"""
    Wrapper class for PyTorch models.
    """
    def __init__(self, model, n_classes,
                 normalize={'mean': [0., 0., 0.], 'std': [1., 1., 1.]},
                 device=None):
        super().__init__()
        
        # Set device
        if device is None:
            try:
                device = next(model.parameters()).device
            except:
                raise ValueError("Please set 'device' argument.")
                
        # Set n_class
        assert isinstance(n_classes, int)
        self.register_buffer('n_classes', torch.tensor(n_classes))

        # Set model structure
        self.norm = Normalize(normalize['mean'], normalize['std']).to(device)
        self.model = model.to(device)
        self.to(device)

    def forward(self, input):
        out = self.norm(input)
        out = self.model(out)
        return out

    # Load from state dict
    def load_dict(self, save_path):
        state_dict = torch.load(save_path, map_location='cpu')
        self.load_state_dict_auto(state_dict['rmodel'])
        print("Model loaded.")

        if 'record_info' in state_dict.keys():
            print("Record Info:")
            print(state_dict['record_info'])

    # DataParallel considered version of load_state_dict.
    def load_state_dict_auto(self, state_dict):
        state_dict = self._convert_dict_auto(state_dict)
        self.load_state_dict(state_dict)

    # Automatically changes pararell mode and non-parallel mode.
    def _convert_dict_auto(self, state_dict):
        keys = state_dict.keys()
                
        save_parallel = any(key.startswith("model.module.") for key in keys)
        curr_parallel = any(key.startswith("model.module.") for key in self.state_dict().keys())
        if save_parallel and not curr_parallel:
            new_state_dict = {k.replace("model.module.", "model."): v for k, v in state_dict.items()}
            return new_state_dict
        elif curr_parallel and not save_parallel:
            new_state_dict = {k.replace("model.", "model.module."): v for k, v in state_dict.items()}
            return new_state_dict
        else:
            return state_dict

    def save_dict(self, save_path):
        save_dict = OrderedDict()
        save_dict["rmodel"] = self.state_dict()
        torch.save(save_dict, save_path)

    def set_parallel(self):
        self = torch.nn.DataParallel(self)
        return self

    def named_parameters_with_module(self):
        module_by_name = {}
        for name, module in self.named_modules():
            module_by_name[name] = module

        for name, param in self.named_parameters():
            if '.' in name:
                module_name = name.rsplit(".", maxsplit=1)[0]
                yield name, param, module_by_name[module_name]
            else:
                yield name, param, None
    
    #################################################
    ############# Evaluate Robustness ###############
    #################################################
    @torch.no_grad()
    def eval_accuracy(self, data_loader):
        return get_accuracy(self, data_loader)

    ##############################################################
    ############# Evaluate Generalization Measures ###############
    ##############################################################
    
    