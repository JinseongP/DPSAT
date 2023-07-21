# DPSAT (Differentially Private Sharpness-Aware Training)

Official PyTorch implementation of "Differentially Private Sharpness-Aware Training", ICML 2023.

We will soon release an update to the official code.


## Environment configuration

The codes are based on python3.8+, CUDA version 11.0+. The specific configuration steps are as follows:

1. Create conda environment
   
   ```shell
   conda create -y -n dpsat python=3.9.7
   conda activate dpsat
   ```

2. Install pytorch (can be different depending on your environments)
   
   ```shell
   conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
   ```

3. Installation profile
   
   ```shell
   pip install -r requirements.txt
   python setup.py develop
   ```
## Data preparation
Firstly, download the datasets used.
- 

```

```
