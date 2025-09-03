# BiVSS-UNet

**BiVSS-UNet** is a deep learning framework for image segmentation, based on the U-Net architecture. The model is implemented in **PyTorch 2.6.0** with **CUDA 12.4** support, and optimized for high-performance GPU inference.

---

## 🚀 Features

- ⚙️ Built with **PyTorch 2.6.0 + CUDA 12.4**
- 🔁 Training pipeline with **PyTorch Lightning**
- 🧪 Extensive support for **evaluation metrics**
- 🧰 Modular and clean codebase for easy extension

---

## 🖥 Environment & Configuration

### ✅ System Summary

| Item              | Info                             |
|-------------------|----------------------------------|
| OS                | Ubuntu                           |
| Python            | ≥ 3.10                           |
| PyTorch           | 2.6.0+cu124                      |
| CUDA Available    | True                             |
| CUDA Version      | 12.4                             |

### 📦 Key Python Packages

Here are the most critical packages used:

- `torch==2.6.0`
- `torchvision==0.21.0`
- `pytorch-lightning==2.2.1`
- `albumentations==1.4.2`
- `opencv-python==4.11.0.86`
- `timm==1.0.15`
- `transformers==4.51.1`
- `tensorboardX==2.6.2.2`
- `einops`, `thop`, `scikit-learn`, `scikit-image`, etc.

You can find the full list in requirements.txt.

---


## 🧪 Usage

### 1. Install

```bash
conda create -n bivss python=3.10
conda activate bivss

conda install -c nvidia cudatoolkit=12.4
conda install -c "nvidia/label/cuda-12.4.0" cuda-nvcc

pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 -f https://download.pytorch.org/whl/torch_stable.html

conda install packaging

cd BiVSS-UNet
pip install -r requirements.txt

cd geoseg/Mamba-UNet/mamba
python setup.py install

cd ../ ; cd causal-conv1d
python setup.py install
```

### 2. Train the model

```bash
bash run_loveda.sh #loveda
```
```bash
bash run_potsdam.sh #potsdam
```
```bash
bash run_vaihingen.sh #vaihingen
```

---


