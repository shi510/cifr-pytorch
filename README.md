# cifr-pytorch (This project is in progress.)
Continuous Implicit Feature Representation

## What's different from LIIF
* Adversarial Training (Generator: Encoder + LIIF, Discriminator: U-Net based)
* An Encoder is StyleGAN based architecture (Noise Injection for generating fine-grained details)  
* Contextual Loss  
* Gradient Normalization for a Discriminator  
* Relativistic Loss (ESRGAN)  
* Positional Encoding of relative coordinates to LIIF (NeRF)  

## 0. Installation
```
apt install ninja-build
pip install -r requirements.txt
```

## 1. Dataset
Download 
[Train Data (HR images)](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) 
and 
[Validation Data (HR images)](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip) 
from 
[DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K).  
```
curl -L http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -o DIV2K_train_HR.zip
curl -L http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip -o DIV2K_valid_HR.zip
```
Decompress thease files to ./sr_dataset folder.  
```
unzip DIV2K_train_HR.zip
unzip DIV2K_valid_HR.zip
```
You have the folder structure.  
```
./sr_dataset
    |
    |-- DIV2K_train_HR
    |
    |-- DIV2K_valid_HR
```

## 2. Train
```
export PYTHONPATH=$(pwd)
python tools/train.py --config configs/div2k_stylegan_sn_liif.py
```

## 3. Inference
```
export PYTHONPATH=$(pwd)
python tools/inference_liif.py \
--config configs/div2k_stylegan_sn_liif.py \
--ckpt work_dir/div2k_stylegan_sn_liif/checkpoints/000200.pth \
--img test.jpg
```

## References
**Really thank the authors of LIIF for sharing their codes and research.**  

[1] Learning Continuous Image Representation with Local Implicit Image Function, Yinbo Chen et al.
