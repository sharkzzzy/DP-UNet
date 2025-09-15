# DP-UNet
**DP-UNet** is a deep learning framework for image segmentation, based on the U-Net architecture. The model is implemented in **PyTorch 2.6.0** with **CUDA 12.4** support, and optimized for high-performance GPU inference.
> _Formerly known as **BiVSS-UNet**._
---
The main model architectures are located in:
👉 ([mamba_sys.py](https://github.com/sharkzzzy/BiVSS-UNet/blob/main/geoseg/models/networks/mamba_sys.py))


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


The folder contains pre-segmented images for the following datasets:

-**LoveDA**

-**ISPRSPotsdam**

-**ISPRSVaihingen**

You can access and download the segmented images here:
👉 ([Download from Google Drive](https://drive.google.com/drive/folders/1CrPBbs1I0oYRvyxqG68q5YX8l-KCLfh5?usp=sharing))

---

### ModelWeights

Download the pretrained weightshere:

-**vmamba_tiny_e292.pth**:
[GoogleDriveLink](https://drive.google.com/file/d/1Vgh0pggmiNdgMswI_t318gGjkPeL6YrT/view?usp=sharing)

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

### 3. Test the model

```bash
bash test_loveda.sh #loveda
```
```bash
bash test_potsdam.sh #potsdam
```
```bash
bash test_vaihingen.sh #vaihingen
```

### 4.Use Trained Model Weights for Inference
```bash
model_weights/.../best.ckpt
```
The trained model weights can be downloaded from the link below:

👉 ([Download from Google Drive](https://drive.google.com/drive/folders/1_plPx7E8LWBu9u8j1IPwmmoZ7pNHPbIY?usp=sharing))

These weights are ready to use for inference or fine-tuning.
You can then use this checkpoint for inference. Make sure your testing script (e.g., loveda_test.py) specifies the path to the checkpoint.
---

## Acknowledgement

Many thanks the following projects's contributions.
- [CM-UNet](https://github.com/XiaoBuL/CM-UNet/tree/main)
- [Mamba-UNet](https://github.com/ziyangwang007/Mamba-UNet)
- [GeoSeg](https://github.com/WangLibo1995/GeoSeg)
