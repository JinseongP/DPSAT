import os
import warnings
import argparse
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import importlib

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import src
import src.trainer as tr

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# Differtial Privacy args
parser.add_argument('--max_grad_norm', type=float, default=0.1, help='gradient clipping paramter C')
parser.add_argument('--epsilon', type=float, default=3.0, help='privacy budget epsilon')
parser.add_argument('--delta', type=float, default=1e-5, help='privacy budget epsilon')

# Optimization args
parser.add_argument('--lr', type=float, default=2, help='learning rate')
parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=2048, help='batch_size')
parser.add_argument('--max_physical_batch_size', type=int, default=1024, help='number of max_physical_batch_size (for distributed training in DP)')
parser.add_argument('--minimizer', type=str, default='DPSAT', help="[None, 'DPSAT' 'DPSATMomentum']")
parser.add_argument('--rho', type=float, default=0.0, help='perturbation radius of sharpness-aware training. rho=0.0 for DPSGD.')

# Dataset and dataloader args
parser.add_argument('--data', type=str, default='CIFAR10', help="['CIFAR10' 'FashionMNIST' 'MNIST' 'SVHN']")
parser.add_argument('--model_name', type=str, default='DPNASNet_CIFAR', help= "['DPNASNet_CIFAR','DPNASNet_FMNIST','DPNASNet_MNIST','Handcrafted_CIFAR','Handcrafted_MNIST','ResNet10']")
parser.add_argument('--normalization', type=bool, default=True)
parser.add_argument('--n_class', type=int, default=10, help='number of classification class')

# Saving args
parser.add_argument('--path', type=str, default="./saved/")
parser.add_argument('--name', type=str, default="saved_name")

# GPU args
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

if __name__ == '__main__':

    args = parser.parse_args()
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        torch.cuda.set_device(args.gpu)
    print('args:', args)

    #### FOR DP
    MAX_GRAD_NORM = args.max_grad_norm
    EPSILON = args.epsilon
    DELTA = args.delta

    #### FOR TRAINING
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    MAX_PHYSICAL_BATCH_SIZE = args.max_physical_batch_size
    RHO = args.rho

    #### FOR SAVE
    PATH = args.path
    NAME = args.name
    SAVE_PATH = PATH + NAME

    ### FOR MODELING
    MODEL_NAME = args.model_name
    DATA = args.data
    if args.normalization==True:
        if args.data == "CIFAR10":
            NORMALIZE = {'mean':[0.4914, 0.4822, 0.4465],
                         'std':[0.2023, 0.1994, 0.2010]}
        elif "MNIST" in args.data: #MNIST, FMNIST
            NORMALIZE = {'mean':[0.1307],
                         'std':[0.3081]}
        elif args.data == "SVHN":
            NORMALIZE = {'mean':[0.4377, 0.4438, 0.4728],
                         'std':[0.1980, 0.2010, 0.1970]}
        else:
            raise NotImplementedError("Choose proper dataset.")
    N_CLASSES = args.n_class

    # ### Data loader
    data = src.Datasets(data_name=DATA, train_transform = transforms.ToTensor())
    train_loader, test_loader = data.get_loader(batch_size=BATCH_SIZE, drop_last_train=False, num_workers=16)

    ### Model & Optimizer
    #### Load model
    model = src.utils.load_model(model_name=MODEL_NAME, n_classes=N_CLASSES).cuda() # Load model
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print("model params: {:.4f}M".format(pytorch_total_params/1000000))

    #### Define optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=LR, momentum=0.9)

    # ### Load PrivacyEngine from Opacus
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=EPOCHS,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    optimizer.target_epsilon = EPSILON
    optimizer.target_delta = DELTA

    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")

    rmodel = src.RobModel(model, n_classes=N_CLASSES, normalize=NORMALIZE).cuda()        

    # ### Start Training
    importlib.reload(tr)

    trainer = tr.DpTrainer(NAME,rmodel)
    trainer.max_physical_batch_size = MAX_PHYSICAL_BATCH_SIZE
    trainer.record_rob(train_loader, test_loader)

    trainer.fit(train_loader=train_loader, max_epoch=EPOCHS, start_epoch=0,
                optimizer=optimizer,
                scheduler=None, scheduler_type="Epoch",
                minimizer="{}(rho={})".format(args.minimizer, RHO),
                save_path=SAVE_PATH, save_best={"Clean(Val)":"HB"},
                save_type=None, save_overwrite=True, record_type="Epoch")

    # ### Evaluation
    rmodel.load_dict(PATH+NAME+'/last.pth')
    rmodel.eval_accuracy(test_loader)

    rmodel.load_dict(PATH+NAME+'/best.pth')
    rmodel.eval_accuracy(test_loader)







