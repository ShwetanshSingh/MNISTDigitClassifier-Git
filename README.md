---
title: MNISTDigitClassifier
app_file: app.py
sdk: gradio
sdk_version: 5.29.1
---
# MNIST Digit Classifier
> [GitHub](https://github.com/ShwetanshSingh/MNISTDigitClassifier-Git)
> [HuggingFace Spaces](https://huggingface.co/spaces/ShwetanshSingh/MNISTDigitClassifier) deployed app

This project is practice for creating a neural network from scratch. The project primarily uses the `Fastai` library. 

## Model
The model is a **single layer network**. The activation function used is a **sigmoid** to make the output of the network as 0 or 1 (digit `3` or `7`).

## Setup
**GitHub Codespaces**
- Create a codespace from code icon at the top
- Run `source setup.sh` in terminal to setup the environment. This process will take time

## Usage
- Run `uv run app.py` in the terminal. The terminal will show the link for the app
- Input the image of a handwritten digit, and submit to get results

After setting up the environment, you should also be able to execute the notebooks in `notebooks/`. `model_from_scratch.ipynb` was used to train and save the model used. 

If you are not familiar with `Fastai` functions, checkout the [Fastai docs](https://docs.fast.ai/). 

`model_using_pytorch_func.ipynb` uses `Pytorch` library instead of `Fastai`.

`sgd_example.ipynb` notebook goes through an example to understand stochastic gradient descent.

## License
[MIT](https://choosealicense.com/licenses/mit/)
