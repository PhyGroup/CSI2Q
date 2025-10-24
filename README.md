# Introduction to CSI²Q

This repository provides the implementation of **CSI²Q**. Please refer to our paper for more details:  Enhancing WiFi CSI Fingerprinting:A Deep Auxiliary Learning Approach


## Setup

### Requirements
- MATLAB R2016b or later (for preprocessing)  
- Python 3.8+  
- PyTorch ≥ 1.12  

## Datasets:

Dataset could be downloaded from this file: [dataset](https://drive.google.com/file/d/1MCMbCFyUGTWum_GKauTNf9gba5llmuCV/view?usp=drive_link](https://drive.google.com/file/d/1MCMbCFyUGTWum_GKauTNf9gba5llmuCV/view?usp=sharing).  
After downloading and extracting, you will see three folders:  

- **IN_lab/** – in-lab CSI dataset (10 routers)  
- **In_wild/** – in-the-wild CSI dataset (25 APs, 12 locations)  
- **wisig/** – synthetic CSI dataset (derived from WiSig IQ)  

Each folder contains two subdirectories:  

- **packets_sltf/** – frequency-domain CSI transformed into 320-point time-domain IQ  
- **packets_sltf_eq/** – results after applying **Channel Interference Mitigation (CIM)** to `packets_sltf`

## Dataset Merging & PKL Generation
Before training the models, we need to further process the  datasets:  
1. **Auxiliary IQ dataset**  
   - The auxiliary IQ dataset is derived from the **WiSig dataset** (open-source IQ samples).  
   - To build this dataset, we first apply **channel equalization** using the MATLAB scripts:  
     - `equalize_channel.m` 
     - `equalize_dataset.m`
   - After equalization, the WiSig IQ data are transformed into 320-point IQ sequences (aligned with CSI representations).  
   - These IQ samples are used as an **independent auxiliary source domain** for auxiliary learning.
     
2. **Concatenate multiple captures of the same AP**  
   - For devices that were collected across multiple sessions/days, we first merge them into a single dataset.  
   - This step is implemented in **MATLAB** (`concatenate_data.m`).

3. **Generate `.pkl` files for PyTorch dataloader**  
   - After merging, convert the processed `.mat` data into `.pkl` format for faster loading in training.  
   - The conversion script is provided in **code/python/** (e.g. `mat_to_pkl.py`).  


  
## Training and Evaluation

> **OWR** = Open-World Recognition
> **MTL_0** = Multi-Task Learning on **WiSig** with source domain
> **MTL** = Multi-Task Learning without source domain

## Citation

Please cite:

@ARTICLE{csi2q_huang,
  author={Huang, Yong and Wang, Wenjing and Zhang, Dalong and Wang, Junjie and Chen, Chen and Cao, Yan and Wang, Wei},
  journal={IEEE Internet of Things Journal}, 
  title={Enhancing WiFi CSI Fingerprinting: A Deep Auxiliary Learning Approach}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Fingerprint recognition;Feature extraction;Wireless fidelity;Radio frequency;Performance evaluation;Time-domain analysis;Interference;Internet of Things;Hardware;Channel estimation;Radio frequency fingerprinting;channel state information;open-world recognition},
  doi={10.1109/JIOT.2025.3625062}}

```bash

cd code/python
# Open-World Recognition
python OWR.py

# Multi-Task Learning without source domain
python MTL.py

# Multi-Task Learning on **WiSig** with source domain
python MTL_0.py  
```
