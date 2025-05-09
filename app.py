from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')
class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']  # Change as needed

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        img = Image.open(file.stream).resize((32, 32))  # Resize as per your model input
        img = np.expand_dims(np.array(img) / 255.0, axis=0)
        preds = model.predict(img)
        predicted_class = class_names[np.argmax(preds)]
        prediction = f'Prediction: {predicted_class}'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
