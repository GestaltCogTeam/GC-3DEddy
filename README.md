# 3D Forecasting of Oceanic Mesoscale Eddies from Satellite Observations with Large Autoregressive Modeling

![GC-3DEddy](assets/GC-3DEddy.jpg)

This repository is built based on [VAR (NeurIPS 2024 Best Paper)](https://github.com/FoundationVision/VAR/)


## Installation

```
conda create -n GC-3DEddy python=3.11
pip install -r requirements.txt
```


## Training Scripts
1. Train VQVAE
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=... train.py --bs=2048 --ep=1000 --fp16=1 --wpe=0.01 --data_path=... --vae_if_train=True --pn=36 --tclip=1.0 --tblr=4e-5 --datasets_name=...
```

2. Train TAT (Thermohaline Autoregressive Transformer)
```
# horizon=5
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=... train.py --depth=16 --bs=32 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.01 --data_path=... --pn=36 --vae_ckpt=... --datasets_name=... --time_patch_num=5 --tblr=1e-3
# horizon=10
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=... train.py --depth=16 --bs=24 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.01 --data_path=... --pn=36_2 --vae_ckpt=... --datasets_name=... --time_patch_num=10 --tblr=1e-3
```


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```

```
