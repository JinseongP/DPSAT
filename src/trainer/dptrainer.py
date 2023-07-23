import os
import warnings
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *

from opacus.utils.batch_memory_manager import BatchMemoryManager

from ._rm import RecordManager
from .. import RobModel
from ..optim import *
from ._trainer import Trainer
from ..utils import get_subloader



r"""
Base class for DP trainers.
"""


class DpTrainer(Trainer):
#     def __init__(self, rmodel):
#         super().__init__("DpTrainer", rmodel)
#         self._flag_record_rob = False
    def __init__(self, name, rmodel):
        super(DpTrainer, self).__init__(name, rmodel)
        self._flag_record_rob = False

    def record_rob(self, train_loader, val_loader,
                   eps=None, alpha=None, steps=None, std=None,
                   n_train_limit=None, n_val_limit=None):
        self._flag_record_rob = True
        self._record_rob_keys = ['Clean']
        self._train_loader_rob = get_subloader(train_loader, n_train_limit)
        self._val_loader_rob = get_subloader(val_loader, n_val_limit)
        self._init_record_keys = self._init_record_keys_with_rob


    def _init_record_keys_with_rob(self, record_type):
        # Add Epoch and Iter to Record_keys
        keys = ["Epoch"]
        if record_type == "Iter":
            keys = ["Epoch", "Iter"]

        # Add Custom Record_dict keys to Record_keys
        for key in self.record_dict.keys():
            keys.append(key)

        # Add robust keys
        keys += [key + '(Tr)' for key in self._record_rob_keys]
        keys += [key + '(Val)' for key in self._record_rob_keys]

        # Add lr to Record_keys
        keys.append("lr")
        return keys

    # Update Records
    def _update_record(self, record, lr):
        if self._flag_record_rob:
            rob_record = []
            for loader in [self._train_loader_rob, self._val_loader_rob]:
                rob_record.append(self.rmodel.eval_accuracy(loader))
            return record + rob_record + [lr]

        return record + [lr]
    
    def fit(self, train_loader, optimizer, max_epoch,
            start_epoch=0, end_epoch=None,
            scheduler=None, scheduler_type=None, minimizer=None,
            save_path=None, save_best=None, save_type="Epoch",
            save_overwrite=False, record_type="Epoch"):

        # Init record and save dicts
        self.rm = None
        self.start_time = None
        self.record_dict = OrderedDict()
        self.curr_record_dict = None
        self.best_record_dict = None
        self.save_dict = OrderedDict()

        # Set Epoch and Iterations
        self.max_epoch = max_epoch
        self.train_loader = train_loader

        # Check Save and Record Values
        self._check_valid_options(save_type)
        self._check_valid_options(record_type)

        # Init Optimizer and Schduler
        self._init_optimizer(optimizer)
        self._init_schdeuler(scheduler, scheduler_type)
        self._init_minimizer(minimizer)


        # Check Save Path
        if save_path is not None:
            if save_path[-1] == "/":
                save_path = save_path[:-1]
            # Save Initial Model
            self._check_path(save_path, overwrite=save_overwrite)
            self._check_path(save_path+"/last.pth", overwrite=save_overwrite, file=True)
            self._check_path(save_path+"/best.pth", overwrite=save_overwrite, file=True)
            self._check_path(save_path+"/record.csv", overwrite=save_overwrite, file=True)
            self._check_path(save_path+"/summary.txt", overwrite=save_overwrite, file=True)

            if save_type in ["Epoch", "Iter"]:
                self._check_path(save_path+"/epoch_iter/", overwrite=save_overwrite)
                self._save_dict_to_file(save_path, 0)
        else:
            raise ValueError("save_path should be given for save_type != None.")

        # Check Save Best
        if save_best is not None:
            if record_type not in ['Epoch', 'Iter']:
                raise ValueError("record_type should be given for save_best = True.")
            if save_path is None:
                raise ValueError("save_path should be given for save_best != None.")

        # Training Start
        for epoch in range(self.max_epoch):
            # Update Current Epoch
            self.epoch = epoch+1

            # If start_epoch is given, update schduler steps.
            if self.epoch < start_epoch:
                if self.scheduler_type == "Epoch":
                    self._update_scheduler()
                elif self.scheduler_type == "Iter":
                    for _ in range(self.max_iter):
                        self._update_scheduler()
                else:
                    pass
                continue

            if end_epoch is not None:
                if self.epoch == end_epoch:
                    break

            self.losses = 0
            self.losses_p = 0
            with BatchMemoryManager(
                data_loader=train_loader, 
                max_physical_batch_size=self.max_physical_batch_size, 
                optimizer=self.optimizer
            ) as memory_safe_data_loader:
                self.max_iter = len(memory_safe_data_loader)
                for i, train_data in enumerate(memory_safe_data_loader):
                    # Init Record and Save dict
                    self._init_record_dict()
                    self._init_save_dict()
                    
                    if self.start_time is None:
                        self.start_time = datetime.now()

                    # Update Current Iteration
                    self.iter = i+1

                    # Set Train Mode
                    self.rmodel = self.rmodel.to(self.device)
                    self.rmodel.train()

                    try:
                        self._do_iter(train_data)
                        warnings.warn(
                            "_do_iter() is deprecated, use _update_weight() instead",
                            DeprecationWarning
                        )
                    except:
                        # Calulate Cost and Update Weight
                        self._update_weight(train_data)

                    if 'DPSAT' not in str(self.minimizer.__class__):
                        self.losses += self.record_dict["CALoss"]
                        self.record_dict["CALoss"] = self.losses/(i+1)
                    
                    if self.minimizer:
                        self.losses_p += self.record_dict["CALoss^p"]
                        self.record_dict["CALoss^p"] = self.losses_p/(i+1)
                    

                    # Print Training Information
                    if record_type is not None:
                        self.rmodel.eval()

                        if self.rm is None:
                            record_keys = self._init_record_keys(record_type)
                            self.rm = RecordManager(record_keys, start_time=self.start_time)
                            print("["+self.name+"]")
                            print("Training Information.")
                            print("-Epochs:", self.max_epoch)
                            print("-Optimizer:", self.optimizer)
                            print("-Scheduler:", self.scheduler)
                            print("-Save Path:", save_path)
                            print("-Save Type:", str(save_type))
                            print("-Record Type:", str(record_type))
                            print("-Device:", self.device)
                            if save_best is not None:
                                self._check_save_best(save_best, record_keys)
                        else:
                            self.rm.progress()

                        iter_record = self._get_record_from_record_dict()
                        if save_type == "Iter":
                            self._save_dict_to_file(save_path, self.epoch, self.iter)
                        if record_type == "Iter":
                            lr = self.optimizer.param_groups[0]['lr']
                            curr_record = self._update_record([self.epoch, self.iter]+iter_record, lr=lr)
                            self._update_record_and_best(record_keys, curr_record,
                                                          save_best, save_path)

                        else:
                            pass
                    else:
                        pass
                
                # Save Model
                if record_type == "Epoch":
                    lr = self.optimizer.param_groups[0]['lr']
                    curr_record = self._update_record([self.epoch]+iter_record, lr=lr)
                    self._update_record_and_best(record_keys, curr_record,
                                                    save_best, save_path)
                if save_type == "Epoch":
                    self._save_dict_to_file(save_path, self.epoch)

                if save_path is not None:
                    self.rm.to_csv(save_path+"/record.csv", verbose=False)
                    self._save_dict_to_file(save_path, is_last=True)

                # Scheduler Step
                if (self.scheduler_type == "Epoch"):
                    self._update_scheduler()
                elif self.scheduler_type == "Iter":
                    self._update_scheduler()
                else:
                    pass
                # print("Number of batches : {}\n".format(i+1))

        # Print Summary
        if record_type is not None:
            self.rm.summary(save_path+"/summary.txt")
            self.rm.to_csv(save_path+"/record.csv", verbose=False)
            self._save_dict_to_file(save_path, is_last=True)
            
    # Update Weight
    def _update_weight(self, *input):
        # SAM, ASAM, GSAM ...
        self.optimizer.zero_grad()
        if self.minimizer is not None:
            if 'DPSAT' not in str(self.minimizer.__class__):
                self.calculate_cost(*input).backward()
            self.minimizer.ascent_step()

            self.calculate_ascent_cost(*input).backward()
            self.minimizer.descent_step()
        else:
            cost = self.calculate_cost(*input)
            cost.backward()
            self.optimizer.step()

    def calculate_ascent_cost(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self.rmodel(images)

        cost = nn.CrossEntropyLoss()(logits, labels)
        self.record_dict["CALoss^p"] = cost.item()
        return cost   
    
    def calculate_cost(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self.rmodel(images)

        cost = nn.CrossEntropyLoss()(logits, labels)
        self.record_dict["CALoss"] = cost.item()
        return cost        
