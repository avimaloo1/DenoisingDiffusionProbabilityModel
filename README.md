# DenoisingDiffusionProbabilityModel

Denoising Diffusion Probabilistic Model (DDPM) - PyTorch

This project implements a simplified Denoising Diffusion Probabilistic Model (DDPM) using PyTorch. The model is trained on a single uploaded image and learns to reconstruct it by progressively denoising random noise.

Overview

Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process.

This implementation:

Adds Gaussian noise to an image over multiple timesteps
Trains a neural network to predict and remove that noise
Generates new images by reversing the diffusion process
Features
Upload and preprocess custom images
Forward diffusion (noise addition)
Reverse diffusion (image generation)
CNN-based encoder-decoder architecture
Mean Squared Error (MSE) training loss
Visualization of original, noisy, and denoised images
Model Architecture

The model is a simple convolutional encoder-decoder:

Encoder
Conv2D → ReLU → MaxPool
Conv2D → ReLU → MaxPool
Conv2D → ReLU
Bottleneck
Fully connected layers
Decoder
ConvTranspose2D + Upsampling
Sigmoid output (image reconstruction)
📂 Project Structure
.
├── ddpm.py            # Main training and inference script
├── README.md          # Project documentation

Getting Started
1. Install Dependencies
pip install torch torchvision matplotlib numpy pillow
2. Run the Code (Google Colab Recommended)
Upload an image when prompted
The model will:
Resize it to 16×16
Train for 200 epochs
Generate denoised samples
Training Details
Parameter	Value
Image Size	16 × 16
Timesteps (T)	300
Epochs	200
Batch Size	100
Learning Rate	0.001
Optimizer	Adam
Loss Function	MSE
Diffusion Process
Forward Process (Noise Addition)

At each timestep:

x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
Reverse Process (Denoising)

The model predicts noise and reconstructs the image step-by-step from pure noise.

Output

The script displays:

Original Image
Noisy Image
Denoised Image (generated)

📚 References
Ho et al., Denoising Diffusion Probabilistic Models (2020)
PyTorch Documentation: https://pytorch.org/
