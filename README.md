# pyroSAE
This is code that implements the Segmentation Autoencoder (SAE) [1] framework as it is used in my
thesis "Segmentation of Ultra-High Field Magnetic Resonance Brain Images for Multi-Parameter Mapping using Deep Learning".

## Setup
1. Get ([anaconda](https://www.anaconda.com/))
2. Create the conda environment
```
conda env create --file environment.yml
```
3. Install this repository as a python package
```
pip install -e .
```

## Training and Inference
1. Run
```
python train_vae.py
```
This will train the vae and report in directory `/checkpoints`.
Data will be downloaded into `/data`.

2. Run
```
python run_tests.py
```
To see some simple results. Most of the experiments in the thesis do not make 
sense in this demo, because validation labels are not available.

## Reference
[1] E. M. Yu et al., An Auto-Encoder Strategy for Adaptive Image Segmentation, 
    Medical Imaging with Deep Learning, 2020