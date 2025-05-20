
This project was developed as part of the **ADA447 - Deep Learning** course.

It uses **FastAI** and **transfer learning** to classify plant diseases in tomato, potato, and pepper leaves using the PlantVillage dataset. Link :  https://www.kaggle.com/datasets/emmarex/plantdisease. The model was trained with different architectures (ResNet34, ResNet50, EfficientNet-B0), and ResNet50 was selected based on performance.

A confidence-based thresholding system was also implemented to detect unknown or out-of-distribution images.

## Features

* 📷 Image classification for 15 plant disease categories
* 🔍 Comparison of multiple CNN architectures
* 🧠 Transfer learning with fine-tuning
* ⚠️ Confidence thresholding for unknown detection
* 🌐 Deployed demo (via Hugging Face Spaces)

## How to Run

Clone the repo and open the notebook in [Google Colab](https://colab.research.google.com/).
