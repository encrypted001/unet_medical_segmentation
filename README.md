# Medical Image Segmentation using U-Net

**Project Author & Maintainer:** Mohit Kumar  
**GitHub:** https://github.com/encrypted001

A modern, dataset-agnostic PyTorch project for binary medical image segmentation. The project provides training, validation, checkpointing, quantitative evaluation, batch inference, and a Streamlit demonstration.

> **Important:** This is a research/educational software project. It is not a clinical diagnostic system. Do not use predictions for medical decisions without appropriate clinical validation.

## ✨ Features

- Clean PyTorch U-Net implementation
- Binary pixel-level segmentation
- Paired image/mask dataset loader
- Resize and lightweight augmentation
- Cross-Entropy + Dice loss
- AdamW optimizer
- Automatic CPU/GPU selection
- Optional CUDA mixed precision
- Best/last model checkpoints
- Dice and IoU evaluation
- Batch prediction CLI
- Streamlit interactive demo
- Docker smoke-test container
- Reproducible random seed

## 📁 Project structure

```text
Medical_Image_Segmentation_Mohit/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── images/
│   └── masks/
├── checkpoints/
├── results/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── metrics.py
│   └── model.py
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── evaluate.py
├── predict.py
├── requirements.txt
├── test_smoke.py
└── train.py
```

## 🚀 Installation

```bash
git clone https://github.com/encrypted001/Medical-Image-Segmentation.git
cd Medical-Image-Segmentation
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🗂️ Dataset format

Images and masks must have matching filenames:

```text
data/
├── images/
│   ├── patient_001.png
│   ├── patient_002.png
│   └── ...
└── masks/
    ├── patient_001.png
    ├── patient_002.png
    └── ...
```

Mask background should be black and the target region white.

Do not commit private, restricted, or identifiable medical data to GitHub.

## 🧠 Train

```bash
python train.py --epochs 30 --batch-size 4 --size 256
```

For a CUDA GPU:

```bash
python train.py --epochs 30 --batch-size 8 --amp
```

The best checkpoint is written to:

```text
checkpoints/best.pt
```

## 📊 Evaluate

```bash
python evaluate.py --model checkpoints/best.pt
```

The result is saved to:

```text
results/metrics.json
```

Metrics:

- Dice coefficient
- Intersection over Union (IoU)
- Number of evaluated samples

Only report metrics produced from your own dataset and experiment.

## 🔎 Prediction

```bash
python predict.py --model checkpoints/best.pt --input data/images/patient_001.png
```

Predictions are saved under `results/predictions/`.

## 🌐 Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

Upload an image and generate its predicted binary segmentation mask.

## 🧪 Smoke test

```bash
python test_smoke.py
```

Expected output:

```text
Smoke test passed: (1, 2, 128, 128)
```

## 🐳 Docker

```bash
docker build -t mohit-medical-segmentation .
docker run --rm mohit-medical-segmentation
```

## 📚 Technical background

The model follows the U-Net encoder-decoder idea introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015).

This repository is a new project implementation and does not include the old `` `hubconf.py` or its pretrained Carvana checkpoint references. The previous codebase used that repository as a foundation; this version separates the project into its own dataset, model, training, evaluation, and application modules.

Upstream reference: https://github.com/

Original paper: https://arxiv.org/abs/1505.04597

## 👤 Author

**Mohit Kumar**  
GitHub: https://github.com/encrypted001

## ⚠️ Disclaimer

This software is intended for learning and research. Segmentation quality depends on the dataset, annotation quality, preprocessing, class definition, and training configuration. The output must not be treated as a diagnosis or as a substitute for qualified medical advice.
