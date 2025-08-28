---
title: MNISTDigitClassifier
app_file: app.py
sdk: gradio
sdk_version: 5.29.1
---
# MNIST Digit Classifier

> [GitHub](https://github.com/ShwetanshSingh/MNISTDigitClassifier-Git)  
> [HuggingFace Spaces](https://huggingface.co/spaces/ShwetanshSingh/MNISTDigitClassifier) deployed app

A binary classifier implemented from scratch to distinguish between handwritten digits 3 and 7 from the MNIST dataset. This project demonstrates fundamental concepts of neural networks and deep learning using a simple architecture.

## Model Architecture

### Overview
- Single layer linear model (shallow network)
- Binary classification (3 vs 7)
- Achieves 96.43% accuracy on validation set
- Implemented using PyTorch for tensor operations
- FastAI for data handling and utilities

### Technical Details
- **Input Layer**: 
  - Dimension: 784 (28x28 flattened images)
  - Preprocessing: Images normalized to [0,1] range
  
- **Model Structure**:
  - Linear transformation: y = weights @ x + bias
  - Weights shape: (784, 1)
  - Bias: Single scalar value
  - Activation: Sigmoid
  
- **Output Interpretation**:
  - Output > 0.5: Classified as digit 3
  - Output ≤ 0.5: Classified as digit 7

## Training Details

### Hyperparameters

- Learning rate: 1.0
- Batch size: 256
- Number of epochs: 20

### Training Process

- **Loss Function**: Custom binary cross-entropy with sigmoid activation
- **Optimizer**: Basic gradient descent
- **Dataset**: MNIST sample dataset (subset of 3s and 7s)
- **Training Strategy**: Manual implementation of:
  - Forward pass
  - Backward pass (backpropagation)
  - Parameter updates
- **Performance**: Achieves 96.43% accuracy on the validation set after 20 epochs of training

## Setup and Installation

### GitHub Codespaces
1. Create a codespace from code icon at the top
2. Run `source setup.sh` in terminal to setup the environment

### Local Installation

1. First, clone the repository
```bash
# Clone the repository
git clone https://github.com/ShwetanshSingh/MNISTDigitClassifier-Git.git
cd MNISTDigitClassifier-Git
```
2. Follow instructions from [uv docs](https://docs.astral.sh/uv/getting-started/installation/) to install uv
3. install packages
```bash
uv sync
```

## Usage

### Using the Web Interface
1. Run the Gradio app:
```bash
uv run app.py
```
2. Open the provided link in your browser
3. Upload an image of a handwritten digit (3 or 7)
4. Click submit to get the prediction

## Project Structure
### Notebooks
- `notebooks/model_from_scratch.ipynb`: Main training notebook with detailed explanations. Uses `Fastai` functions
- `notebooks/model_using_pytorch_func.ipynb`: Alternative implementation using PyTorch
- `notebooks/sgd_example.ipynb`: Tutorial on stochastic gradient descent

### Key Files
- `app.py`: Gradio web interface for predictions
- `setup.sh`: Environment setup script
- `models/mnist_model.pkl`: Saved model weights

## Implementation Details

The project demonstrates several key machine learning concepts:
- Basic tensor operations
- Manual gradient descent implementation
- Custom loss function design
- Data preprocessing and normalization
- Model serialization and loading
- Web interface development

## Resources
- [FastAI Documentation](https://docs.fast.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## License
[MIT](https://choosealicense.com/licenses/mit/)
