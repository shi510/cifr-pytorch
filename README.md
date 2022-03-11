# cifr-pytorch (This project is in progress.)
Continuous Implicit Feature Representation

# 0. Installation
```
apt install ninja-build
pip install -r requirements.txt
```

# 1. Dataset
Download 
[Train Data (HR images)](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) 
and 
[Validation Data (HR images)](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip) 
from 
[DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K).  
Decompress thease files to ./sr_dataset folder.  
You have the folder structure.  
```
./sr_dataset
    |
    |-- DIV2K_train_HR
    |
    |-- DIV2K_valid_HR
```

# 2. Train
```
python tools/train.py --config configs/div2k_stylegan_gn_liif.py
```
