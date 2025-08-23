import gradio as gr
import torch
import pickle
from PIL import Image
import numpy as np
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model(filename="./models/mnist_model.pkl"):
    try:
        logging.info(f"Attempting to load model from {filename}")
        with open(filename, "rb") as f:
            model_params = pickle.load(f)

        # Convert numpy arrays back to tensors and set requires_grad
        loaded_weights = torch.tensor(model_params["weights"], requires_grad=True)
        loaded_bias = torch.tensor(model_params["bias"], requires_grad=True)

        logging.info("Model loaded successfully")
        return loaded_weights, loaded_bias
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        logging.error(traceback.format_exc())
        raise


# Load the model
weights, bias = load_model()


def linear1(xb):
    return xb @ weights + bias


def predict(input_image):
    try:
        logging.info("Starting prediction")
        if input_image is None:
            return "Please provide an image"

        # Convert to PIL Image if needed
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        # Preprocess the image
        img = input_image.convert("L")  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to match MNIST format

        # Convert to tensor and normalize
        img_tensor = torch.tensor(np.array(img)).float() / 255
        img_tensor = img_tensor.view(-1, 28 * 28)  # Reshape to match model input

        # Get prediction
        out = linear1(img_tensor)
        prediction = 3 if out.item() > 0.5 else 7
        confidence = torch.sigmoid(out).item()
        confidence = confidence if prediction == 3 else 1 - confidence

        result = f"Predicted Digit: {prediction}\nConfidence: {confidence:.2%}"
        logging.info(f"Prediction complete: {result}")
        return result

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        return error_msg


# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload Image",
        image_mode="L",  # Accept grayscale images
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="MNIST Digit Classifier (3 vs 7)",
    description="Upload an image of a handwritten digit (3 or 7) and the model will predict which digit it is.",
    examples=[
        # You can add example image paths here if you have any
    ],
)

if __name__ == "__main__":
    logging.info("Starting Gradio interface")
    try:
        iface.launch(debug=True)
        logging.info("Gradio interface launched successfully")
    except Exception as e:
        logging.error(f"Failed to launch Gradio interface: {str(e)}")
        logging.error(traceback.format_exc())
        raise
