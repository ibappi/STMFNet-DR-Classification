# STMFNet: Spatial Texture Multi-Scale Feature Fusion Attention Network for Diabetic Retinopathy Classification

This repository contains the implementation of **STMFNet**, a hierarchical deep learning framework designed for diabetic retinopathy (DR) classification using retinal fundus images.

The proposed method combines texture spatial attention mechanisms with multi-scale feature fusion to improve the detection of subtle retinal abnormalities such as microaneurysms, hemorrhages, and abnormal blood vessel patterns.

---

## Paper Information

**Title:** STMFNet: Spatial Texture Multi-Scale Feature Fusion Attention Network for Diabetic Retinopathy Classification

**Authors:** MD Ilias Bappi, Md Monir Ahammod Bin Atique, Kyungbaek Kim

**Conference:** 2025 International Conference on Artificial Intelligence in Information and Communication (ICAIIC)

**Publisher:** IEEE

**Pages:** 0504–0509

**Publication Date:** 2025-02-18

---

## Overview

Diabetic Retinopathy (DR) is one of the leading causes of blindness among diabetic patients worldwide. Early and accurate diagnosis is critical for preventing severe vision loss.

Traditional CNN-based approaches often struggle to capture subtle retinal lesions and texture-related abnormalities. To address this challenge, we propose **STMFNet**, which integrates:

- Texture Spatial Attention Network
- Multi-Scale Feature Fusion
- EfficientNet Backbone
- Hierarchical Feature Learning

The proposed framework improves feature representation while suppressing irrelevant background information.

---

## Key Contributions

- Texture-aware spatial attention mechanism for retinal lesion localization
- Multi-scale feature fusion for capturing fine-grained retinal patterns
- EfficientNet-based hierarchical feature extraction
- Improved DR severity classification performance
- Robust detection of subtle retinal abnormalities
---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/STMFNet-Diabetic-Retinopathy-Classification.git
cd STMFNet-Diabetic-Retinopathy-Classification
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

```text
data/
└── DR_dataset/
    ├── No_DR/
    ├── Mild/
    ├── Moderate/
    ├── Severe/
    └── Proliferative_DR/
```

---

## Results

The proposed STMFNet model demonstrated strong performance in diabetic retinopathy classification by effectively learning texture-sensitive and multi-scale retinal representations.

| Model | Performance |
|---|---|
| STMFNet | Improved DR Classification Accuracy |

---

## Methodology

The proposed framework consists of:

1. EfficientNet backbone for hierarchical feature extraction
2. Texture Spatial Attention module for lesion-focused learning
3. Multi-Scale Feature Fusion block
4. Fully connected classification head

The attention mechanism helps focus on disease-related retinal regions while minimizing irrelevant background information.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{bappi2025stmfnet,
  title={STMFNet: Spatial Texture Multi-Scale Feature Fusion Attention Network for Diabetic Retinopathy Classification},
  author={Bappi, MD Ilias and Atique, Md Monir Ahammod Bin and Kim, Kyungbaek},
  booktitle={2025 International Conference on Artificial Intelligence in Information and Communication (ICAIIC)},
  pages={504--509},
  year={2025},
  publisher={IEEE}
}
```

---

## Authors

- MD Ilias Bappi
- Md Monir Ahammod Bin Atique
- Kyungbaek Kim

---

## License

This repository is intended for academic and research purposes.
