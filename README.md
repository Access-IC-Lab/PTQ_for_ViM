<div align="center">
<h1>PTQ for ViM </h1>
<h3>PTQ for ViM with k-Scaled Quantization and Reparameterization</h3>

Bo-Yun Shi, Yi-Cheng Lo, An-Yeu (Andy) Wu, and Yi-Min Tsai

Paper: ([arXiv 2501.16738](https://arxiv.org/abs/2501.16738))

</div>

## Abstract
In this work, we focus on the post-training quantization (PTQ) of Vision Mamba. We address the issues with three core techniques: 1) a k-scaled token-wise quantization method for linear and convolutional layers, 2) a reparameterization technique to simplify hidden state quantization, and 3) a factor-determining method that reduces computational overhead by integrating operations. Through these methods, the error caused by PTQ can be mitigated. Experimental results on ImageNet-1k demonstrate only a 0.8–1.2\% accuracy degradation due to PTQ, highlighting the effectiveness of our approach.


## Getting Started

### Create Environment (using conda)

```bash
conda create -n {env_name} python=3.10.3
conda activate {env_name}
```

### Install torch + cuda toolkit

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
```

### Download ViM Files from GitHub

Clone the repository:

```bash
git clone https://github.com/Access-IC-Lab/PTQ_for_ViM.git
cd PTQ_for_ViM
```

Alternatively, manually download the zip file from:  
[https://github.com/Access-IC-Lab/PTQ_for_ViM](https://github.com/Access-IC-Lab/PTQ_for_ViM)

### Install Required Packages

```bash
pip install -r requirements.txt
```

## Inference Model

### Running Quantization and Inference

To run calibration for quantization and inference the quantized model:
```shell
python main.py --eval --resume {path_to_pretrained_model.pth} --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path {path_to_dataset} --device cuda --batch-size 128 --quantization-config PTQ4ViM --calibration-size 256 --quantization
```

### Running Using Scripts

Run FP inference on ViM tiny/small model on a specific GPU (replace `{t/s}` with `t` or `s`, and `{GPU}` with the GPU number):

```bash
source scripts/eval-pt-{t/s}.sh {GPU}
```

Run quantization on ViM tiny/small model on a specific GPU:

```bash
source scripts/q-eval-pt-{t/s}.sh {GPU}
```

### Quantization Configuration Setting

QuantConfig class can be found in `quant_config/QuantConfig.py`. Basic setting for this work is defined in `quant_config/PTQ4ViM.py`.


## Acknowledgement

This project is based on Mamba([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Vision Mamba([paper](https://arxiv.org/abs/2401.09417), [code](https://github.com/hustvl/Vim)), and the quantization framework is partially adopted from PTQ4ViT([code](https://github.com/hahnyuan/PTQ4ViT)). Thanks for their excellent works.

