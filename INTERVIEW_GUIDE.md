# Interview Guide

## 60-second explanation

I built this project to compare classical machine-learning and neural-network approaches on real handwritten-digit data while keeping model selection separate from final testing. I used the UCI digits dataset through scikit-learn, created train/validation/test splits, compared baselines and learned models, and exported model parameters for browser and Python inference. The seed-42 neural result is 98.06% accuracy on 360 held-out images, but logistic regression reaches the same accuracy, so I do not claim the neural network is automatically better. I also added a PyTorch implementation and robustness tests so I can discuss framework-level training and failure behavior rather than only clean-test accuracy.

## Questions I should be able to answer

- Why keep validation and test separate?
- Why can a simpler logistic model match the MLP here?
- What does macro F1 add beyond accuracy?
- How does the PyTorch training loop work: DataLoader, forward pass, CrossEntropyLoss, backpropagation, AdamW, early stopping?
- What happens under blur, noise, or occlusion?
- How are the exported weights used by the demo/API?
- What would I change for a larger image problem?

## Strong limitation statement

This is a small 8x8 benchmark. It demonstrates experimental discipline, PyTorch fundamentals, inference, testing, and reproducibility; it does not demonstrate large-scale vision, transformers, CUDA optimization, or distributed training.

## 20-second recruiter version

Real-data classification project with train/validation/test discipline, classical baselines, neural networks, PyTorch, robustness testing, an inference API, browser parity checks, Docker, and CI. The main lesson is model judgment: a neural model achieved 98.06% on the seed-42 holdout, but the simpler logistic reference matched it.
