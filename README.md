# 🧠 Image Classifier with Flask + TensorFlow

An AI-powered image classifier web app built using **TensorFlow**, **Flask**, and **TailwindCSS**. It classifies images using a CNN model trained on CIFAR-10.

![Model Accuracy](https://raw.githubusercontent.com/Mdbaizidtanvir/image-classifier/refs/heads/main/accracy.png)

---

## 🚀 Features

- Upload any image and get a classification prediction
- Trained on the CIFAR-10 dataset
- Clean and responsive UI using TailwindCSS
- Easy to deploy on platforms like **Render** or **Heroku**

---

## 🛠 Tech Stack

- Python 3.10+
- TensorFlow / Keras
- Flask
- Tailwind CSS
- Gunicorn (for deployment)

---

## 📦 Installation

```bash
git clone https://github.com/Mdbaizidtanvir/image-classifier.git
cd image-classifier

# Create virtual env
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
