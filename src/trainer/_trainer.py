import os
import warnings
from collections import OrderedDict
from datetime import datetime
from copy import deepcopy 

import torch
from torch.optim import *
from torch.optim.lr_scheduler import *

from ._rm import RecordManager
from .. import RobModel
from ..optim import *


r"""
Base class for all trainers.

"""


class Trainer():
    def __init__(self, name, rmodel, device=None):
        assert isinstance(rmodel, RobModel)
        self.name = name
        self.rmodel = rmodel
        if device is None:
            self.device = next(rmodel.parameters()).device
        else:
            device = device

    def fit(self, train_loader, optimizer, max_epoch,
            start_epoch=0, end_epoch=None,
            scheduler=None, scheduler_type=None, minimizer=None,
            save_path=None, save_best=None, save_type="Epoch",
            save_overwrite=False, record_type="Epoch", clip_grad_norm=None):

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
        self.max_iter = len(train_loader)

        # Check Save and Record Values
        self._check_valid_options(save_type)
        self._check_valid_options(record_type)

        # Init Optimizer and Schduler
        self._init_optimizer(optimizer)
        self._init_schdeuler(scheduler, scheduler_type)
        self._init_minimizer(minimizer)
        
        self.clip_grad_norm = clip_grad_norm

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
                    

            for i, train_data in enumerate(train_loader):
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

                # Check Last Batch
                is_last_batch = (self.iter == self.max_iter)
                self.rmodel.eval()

                # Print Training Information
                if record_type is not None:
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
                    if record_type == "Epoch" and is_last_batch:
                        lr = self.optimizer.param_groups[0]['lr']
                        curr_record = self._update_record([self.epoch]+iter_record, lr=lr)
                        self._update_record_and_best(record_keys, curr_record,
                                                      save_best, save_path)
                    elif record_type == "Iter":
                        lr = self.optimizer.param_groups[0]['lr']
                        curr_record = self._update_record([self.epoch, self.iter]+iter_record, lr=lr)
                        self._update_record_and_best(record_keys, curr_record,
                                                      save_best, save_path)
                    else:
                        pass

                    if save_path is not None:
                        self.rm.to_csv(save_path+"/record.csv", verbose=False)
                        self._save_dict_to_file(save_path, is_last=True)
                else:
                    pass

                # Save Model
                if save_type == "Epoch" and is_last_batch:
                    self._save_dict_to_file(save_path, self.epoch)
                elif save_type == "Iter":
                    self._save_dict_to_file(save_path, self.epoch, self.iter)
                else:
                    pass

                # Scheduler Step
                if (self.scheduler_type == "Epoch") and is_last_batch:
                    self._update_scheduler()
                elif self.scheduler_type == "Iter":
                    self._update_scheduler()
                else:
                    pass

        # Print Summary
        if record_type is not None:
            self.rm.summary(save_path+"/summary.txt")
            self.rm.to_csv(save_path+"/record.csv", verbose=False)
            self._save_dict_to_file(save_path, is_last=True)

    ################################
    # CAN OVERRIDE BELOW FUNCTIONS #
    ################################

    # Calulate Cost
    def calculate_cost(self, *input):
        raise NotImplementedError

    def _init_record_keys(self, record_type):
        # Add Epoch and Iter to Record_keys
        keys = ["Epoch"]
        if record_type == "Iter":
            keys = ["Epoch", "Iter"]

        # Add Custom Record_dict keys to Record_keys
        for key in self.record_dict.keys():
            keys.append(key)

        # Add lr to Record_keys
        keys.append("lr")
        return keys

    def _update_record(self, record, lr=None):
        # Add Epoch and Iter to Record_keys
        return record + [lr]

    # Update Weight
    def _update_weight(self, *input):
        # SAM, ASAM, GSAM ...
        if self.minimizer is not None:
            self.calculate_cost(*input).backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.rmodel.parameters(), self.clip_grad_norm)
            self.minimizer.ascent_step()
            
            # BridgedSAM
            if hasattr(self.minimizer, 'middle_step'):
                self.calculate_cost(*input).backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.rmodel.parameters(), self.clip_grad_norm)
                self.minimizer.middle_step()

            self.calculate_cost(*input).backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.rmodel.parameters(), self.clip_grad_norm)
            self.minimizer.descent_step()
        else:
            self.optimizer.zero_grad()
            cost = self.calculate_cost(*input)
            cost.backward()
            self.optimizer.step()

    # Update scheduler
    def _update_scheduler(self):
        self.scheduler.step()

    ####################################
    # DO NOT OVERRIDE BELOW FUNCTIONS #
    ###################################

    # Get records from record_dict
    def _get_record_from_record_dict(self):
        iter_record = []
        for key in self.record_dict.keys():
            value = self.record_dict[key]
            if isinstance(value, torch.Tensor):
                value = value.item()
            iter_record.append(value)
        return iter_record

    def _init_record_dict(self):
        del self.record_dict
        self.record_dict = OrderedDict()

    def _init_save_dict(self):
        del self.save_dict
        self.save_dict = OrderedDict()

    # Init Optimizer
    def _init_optimizer(self, optimizer):
        if not isinstance(optimizer, str):
            self.optimizer = optimizer
        else:
            exec("self.optimizer = " + optimizer.split("(")[0] + "(self.rmodel.parameters()," + optimizer.split("(")[1])

    # Init Scheduler
    def _init_schdeuler(self, scheduler, scheduler_type):
        if not isinstance(scheduler, str):
            if scheduler is None:
                self.scheduler = None
                self.scheduler_type = None
            else:
                if scheduler_type is None:
                    raise ValueError("The type of scheduler must be specified as 'Epoch' or 'Iter'.")
                self.scheduler = scheduler
                self.scheduler_type = scheduler_type
        else:
            if "Step(" in scheduler:
                # Step(milestones=[2, 4], gamma=0.1)
                exec("self.scheduler = MultiStepLR(self.optimizer, " + scheduler.split("(")[1])
                self.scheduler_type = 'Epoch'

            elif 'Cyclic(' in scheduler:
                # Cyclic(base_lr=0, max_lr=0.3)
                lr_steps = self.max_epoch * self.max_iter
                exec("self.scheduler=CyclicLR(self.optimizer, " + scheduler.split("(")[1].split(")")[0] + \
                     ", step_size_up=lr_steps/2, step_size_down=lr_steps/2)")
                self.scheduler_type = 'Iter'

            elif 'Cosine' == scheduler:
                # Cosine
                self.scheduler = CosineAnnealingLR(self.optimizer, self.max_epoch, eta_min=0)
                self.scheduler_type = 'Epoch'

            else:
                exec("self.scheduler = " + scheduler.split("(")[0] + "(self.optimizer, " + scheduler.split("(")[1])
                self.scheduler_type = scheduler_type

    # Init Minimizer
    def _init_minimizer(self, minimizer):
        if not isinstance(minimizer, str):
            self.minimizer = minimizer
        else:
            exec("self.minimizer = " + minimizer.split("(")[0] + "(self.optimizer, self.rmodel," + minimizer.split("(")[1])

    # Check whether save best is possible
    def _check_save_best(self, save_best, record_keys):
        for key in save_best:
            if key not in record_keys:
                raise ValueError("Check save_best: the key must be in records_keys.")

        self._LBS = []
        self._HBS = []
        self._LBOS = []
        self._HBOS = []
        for key, judge in save_best.items():
            if judge == "LB":
                self._LBS.append(key)
            elif judge == "HB":
                self._HBS.append(key)
            elif judge == "LBO":
                self._LBOS.append(key)
            elif judge == "HBO":
                self._HBOS.append(key)
            else:
                raise ValueError("Values of save_best should be in ['LB', 'HB' ,'LBO', 'HBO'].")

    # Check the current is the best
    def _check_record_best(self, save_best):
        if self.best_record_dict is not None:
            flag_tie = True
            # Lower is better
            for key in self._LBS:
                best_value = self.best_record_dict[key]
                curr_value = self.curr_record_dict[key]
                if best_value < curr_value:
                    return False
                elif best_value > curr_value:
                    flag_tie = False
                else:
                    pass

            # Higher is better
            for key in self._HBS:
                best_value = self.best_record_dict[key]
                curr_value = self.curr_record_dict[key]
                if best_value > curr_value:
                    return False
                elif best_value < curr_value:
                    flag_tie = False
                else:
                    pass

            # If tie, go for options
            if flag_tie is True:
                # Lower is better
                for key in self._LBOS:
                    best_value = self.best_record_dict[key]
                    curr_value = self.curr_record_dict[key]
                    if best_value < curr_value:
                        return False

                # Higher is better
                for key in self._HBOS:
                    best_value = self.best_record_dict[key]
                    curr_value = self.curr_record_dict[key]
                    if best_value > curr_value:
                        return False

        self.best_record_dict = self.curr_record_dict
        return True

    def _update_record_and_best(self, record_keys, curr_record,
                                 save_best, save_path):
        self.curr_record_dict = OrderedDict(zip(record_keys, curr_record))
        self.rm.add(curr_record)

        if save_best is not None:
            is_best = self._check_record_best(save_best)
            if is_best:
                self._save_dict_to_file(save_path, is_best=True)

    # Check and Create Path
    def _check_path(self, path, overwrite=False, file=False):
        if os.path.exists(path):
            if overwrite:
                print("Warning! Save file(s) will be overwritten:" + path)
            else:
                raise ValueError('[%s] is already exists.'%(path))
        else:
            if not file:
                os.makedirs(path)

    # Check Valid Options
    def _check_valid_options(self, key):
        if key in ["Epoch", "Iter", None]:
            pass
        else:
            raise ValueError(key, " is not valid. [Hint:'Epoch', 'Iter', None]")

    def _save_dict_to_file(self, save_path,
                            epoch=0, i=0,
                            is_last=False, is_best=False):

        self.save_dict["rmodel"] = self.rmodel.cpu().state_dict()
        if self.curr_record_dict is not None:
            self.save_dict["record_info"] = self.curr_record_dict

        if is_last:
            torch.save(self.save_dict, save_path+"/last.pth")
        elif is_best:
            torch.save(self.save_dict, save_path+"/best.pth")
        else:
            torch.save(self.save_dict,
                       save_path+"/epoch_iter/"+str(epoch).zfill(len(str(self.max_epoch)))\
                       +"_"+str(i).zfill(3)+".pth")

        self.rmodel = self.rmodel.to(self.device)

