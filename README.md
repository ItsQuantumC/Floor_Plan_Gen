# 🏠 Floor Plan Generation using Deep Generative Models

This project explores the use of **deep generative models** for the task of **automated architectural floorplan generation**.  
We experiment with three different types of Generative Adversarial Networks (GANs):

- 🎲 **DCGAN** (Deep Convolutional GAN)
- ✍️ **Pix2Pix** (Conditional GAN for Image-to-Image Translation)
- 🧠 **SAGAN** (Self-Attention GAN for capturing long-range dependencies)

Each model is trained and evaluated on rasterized floorplan datasets (primarily **CubiCasa5K** and **ROBIN**).  
Different preprocessing techniques were employed depending on the model architecture, ranging from **noise vector generation** to **edge map conditioning** and **attention-based feature extraction**.

---

## ⚙️ Experiments Conducted

- Training **DCGAN** to synthesize floorplans purely from random noise vectors.
- Training **Pix2Pix** using edge maps to translate to realistic floorplans.
- Training **SAGAN** with self-attention layers to capture global spatial coherence.
- Computing evaluation metrics such as **Structural Similarity Index (SSIM)** and **Peak Signal-to-Noise Ratio (PSNR)** for Pix2Pix outputs.

---

## 🖼️ Example Results

### DCGAN

<img src="assets/fake_epoch_100.png" width="450"/>

---

### SAGAN
> **Attention-driven floorplan generation**

<img src="assets/fixed_epoch100.png" width="450"/>

---

### Pix2Pix

<img src="assets/fake_49.png" width="450"/>

---
