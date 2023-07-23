# DPSAT (Differentially Private Sharpness-Aware Training, ICML 2023)

**This is an official PyTorch implementation of DPSAT: [Differentially Private Sharpness-Aware Training](https://arxiv.org/abs/2306.05651).**


## 1. Environment configuration

The code is based on Python 3.8+ and requires CUDA version 11.0 or higher. Follow the specific configuration steps below to set up the environment: 

1.  Create a conda environment:
   
   ```shell
   conda create -y -n dpsat python=3.9.7
   conda activate dpsat
   ```

2. Install PyTorch (version can vary depending on your environment):
   
   ```shell
   conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
   ```

3. Install the required packages:
   
   (Note: opacus==1.2.0 is not enforced in requirements as it may update the torch version to 2.0. Instead, we use the codes in the opacus folder.):
   
   ```shell
   pip install -r requirements.txt
   ```
## 2. Run py code

### Hyperparameters:

- Differtial Privacy args

  - max_grad_norm: default=0.1, help='gradient clipping paramter C'

  - epsilon: default=3.0, help='privacy budget epsilon'

  - delta: default=1e-5, help='privacy budget epsilon'

- Optimization args

  - lr: default=2, help='learning rate'
  - epochs: default=30, help='number of training epochs'
  - batch_size: default=2048, help='batch_size'
  - max_physical_batch_size: default=1024, 
    help='number of max_physical_batch_size (for distributed training in DP)'
  - minimizer: default='DPSAT',  help="[None, 'DPSAT' 'DPSATMomentum']"
  - rho: default=0.0, help='perturbation radius of sharpness-aware training. **rho=0.0 for DPSGD**.'
  
- Dataset and dataloader args

  - data: default='CIFAR10', help="['CIFAR10' 'FashionMNIST' 'MNIST' 'SVHN']")

  - model_name: default='DPNASNet_CIFAR', 
    help= "['DPNASNet_CIFAR:'DPNASNet_FMNIST:'DPNASNet_MNIST:'Handcrafted_CIFAR:'Handcrafted_MNIST:'ResNet10']"

  - normalization: default=True
  - n_class: default=10, help='number of classification class'

- Saving args

  - path: default="./saved/"

  - name: default="saved_name"

- GPU args

  - use_gpu: default=True, help='use gpu'

  - gpu: default=0, help='gpu'

### Run for DPSAT:

Set the above parameters in args. To train DPSAT, you should set rho > 0 (rho = 0.0 works the same as DPSGD). 

```shell
python main.py --rho 0.01
```

## 3. Run ipynb code
The trainer is designed to work seamlessly with the ipynb kernel, making it easy to use within Jupyter notebooks. 

**For usage examples, please refer to the `/examples/` folder in this repository.**


## 4. Citation
```
@article{park2023differentially,
  title={Differentially Private Sharpness-Aware Training},
  author={Park, Jinseong and Kim, Hoki and Choi, Yujin and Lee, Jaewook},
  journal={arXiv preprint arXiv:2306.05651},
  year={2023}
}
```

## 5. Reference

- The backbone trainer architecture of this code is based on [adversarial-defenses-pytorch](https://github.com/Harry24k/adversarial-defenses-pytorch) by the co-author, Hoki Kim. For better usage of the trainer, please refer to adversarial-defenses-pytorch.

- Furthermore, we use [Opacus](https://github.com/pytorch/opacus) version 1.2.0 to ensure differentially private training.
- For model architectures, refer to [DPNAS](https://github.com/TheSunWillRise/DPNAS) and [Handcrafted-DP](https://github.com/ftramer/Handcrafted-DP).