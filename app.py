from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load model
model = tf.keras.models.load_model("model/cifar10_classifier.h5")

# Class names (CIFAR-10)
class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Initialize Flask
app = Flask(__name__)

# HTML interface
HTML = '''
<!doctype html>
<title>CIFAR-10 Image Classifier</title>
<h1>Upload an image (32x32)</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file><br><br>
  <input type=submit value=Upload>
</form>
{% if prediction %}
  <h2>Prediction: {{ prediction }}</h2>
{% endif %}
'''

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            img = Image.open(file).resize((32, 32)).convert("RGB")
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            preds = model.predict(img_array)
            prediction = class_names[np.argmax(preds)]
    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
