# DCBP
This is a repository about the paper "Dual-field-of-view Context Aggregation and Boundary Perception for Airport Runway Extraction", accepted to IEEE TGRS 2023.
(https://ieeexplore.ieee.org/document/10113367)

## Prerequisites

> opencv-python==4.1.1  
  pytorch==1.11.0  
  torchvision==0.12.0  
  
## Get Started
### Data Preprocessing
You can download the ARS dataset and pretrained model at (https://pan.baidu.com/s/1q8yuWSESqekpPAKdGFApyw), code:nxnu.
### Model Training
To train a model from scratch, use
```bash
python train.py
```
### Model Evaluation
To evaluate a model, use
```bash
python test_eva.py
```

## License
This project is released under the [Unlicense](/LICENSE).
