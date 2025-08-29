# 🌟 SRCNN Super-Resolution Implementation

A simple implementation of **Super-Resolution Convolutional Neural Network (SRCNN)** that transforms low-resolution images into high-resolution images using deep learning.

## 🔍 What is SRCNN?

SRCNN is a pioneering deep learning approach for single image super-resolution that:
- Uses a neural network to convert low-resolution images to high-resolution
- Employs 3 convolutional layers for feature extraction and reconstruction
- Takes an upscaled low-resolution image (via bicubic interpolation) as input and enhances it using CNN

## 🚀 Features

- **Simple Architecture**: 3-layer CNN with optimized kernel sizes
- **End-to-End Learning**: Direct mapping from low-res to high-res images
- **Quality Metrics**: PSNR (Peak Signal-to-Noise Ratio) evaluation
- **Visual Comparison**: Side-by-side comparison of original, upscaled, and SRCNN outputs

## 📋 Requirements

```bash
tensorflow>=2.0
keras
numpy
opencv-python
matplotlib
scikit-image
```

## 🏗️ Model Architecture

| Layer | Type | Filters | Kernel Size | Activation |
|-------|------|---------|-------------|------------|
| 1 | Conv2D | 64 | 9×9 | ReLU |
| 2 | Conv2D | 32 | 1×1 | ReLU |
| 3 | Conv2D | 1 | 5×5 | Linear |

## 📊 Implementation Steps

### 1. Data Preparation
- Load high-resolution image (128×128)
- Create low-resolution version (32×32) using cubic interpolation
- Upscale low-res back to 128×128 as model input
- Convert to Y channel (luminance) for processing

### 2. Model Training
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Training**: Direct mapping from upscaled to original image

### 3. Evaluation
- Visual comparison between original, upscaled, and SRCNN output
- PSNR calculation for quantitative quality assessment

## 🎯 Results

The model demonstrates improvement in image quality:
- **Upscaled PSNR**: Baseline bicubic interpolation quality
- **SRCNN PSNR**: Enhanced quality after neural network processing

## 📁 Project Structure

```
├── Day21_SRCNN_SuperResolution_Hinglish.ipynb
├── README.md
└── lena.jpg (automatically downloaded)
```

## 🔧 Usage

1. **Clone or download** the notebook
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the notebook** cell by cell
4. **View results** in the generated plots

The notebook automatically downloads the Lena test image if not present.

## 📈 Key Metrics

- **Input Size**: 128×128×1 (Y channel)
- **Scale Factor**: 4x (32×32 → 128×128)
- **Training Epochs**: 100
- **Evaluation Metric**: PSNR (dB)

## 🎨 Sample Output

The implementation provides visual comparisons showing:
- Original high-resolution image
- Low-resolution input
- Bicubic upscaled baseline
- SRCNN enhanced result

## 🔬 Technical Details

- **Color Space**: YCrCb (processing Y channel only)
- **Normalization**: Pixel values scaled to [0,1] range
- **Architecture**: Inspired by the original SRCNN paper
- **Training Strategy**: Single image overfitting for demonstration

## 🚀 Future Improvements

- **Dataset Expansion**: Train on multiple images
- **Advanced Architectures**: Implement ESPCN, SRGAN, or EDSR
- **Multi-scale Training**: Handle different scaling factors
- **Color Enhancement**: Process all color channels

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📞 Connect with Me

**LinkedIn**: [https://www.linkedin.com/in/yash-pal-since2004/]

---

## 📚 References

- Dong, C., et al. "Learning a Deep Convolutional Network for Image Super-Resolution" (ECCV 2014)
- Original SRCNN implementation concepts

---

**⭐ If you found this helpful, please give it a star!**