from fastai.vision.all import *
import gradio as gr
from PIL import Image

# Load the trained model
learn = load_learner("resnet50_best.pkl")

# Set threshold value
THRESHOLD = 0.8

def predict(img):
    try:
        
        img = PILImage.create(img)

        pred_class, _, outputs = learn.predict(img)
        confidence = outputs.max().item()

        if confidence < THRESHOLD:
            return "Unknown"
        return str(pred_class)

    except Exception as e:
        return f"⚠️ Error: {e}"


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Plant Disease Classifier",
    description="Classifies plant leaf diseases for tomato, potato and pepper. If model is not confident enough, it returns 'Unknown'."
).launch()
