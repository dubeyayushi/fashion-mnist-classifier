from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO

app = Flask(__name__)

# Load trained CNN model
cnn_model = load_model("cnn_fashion_model.keras")

# Fashion-MNIST class names
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

@app.route("/")
def home():
    return render_template("index.html")  # webpage to upload/test images

@app.route("/health")
def health():
    return "ok", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Received /predict request")
        file = request.files["file"]
        print("File received")
        img = image.load_img(BytesIO(file.read()), target_size=(28, 28), color_mode="grayscale")
        print("Image loaded with keras")
        img_array = image.img_to_array(img) / 255.0
        img_input = np.expand_dims(img_array, axis=0)
        print("Image preprocessed")

        preds = cnn_model.predict(img_input)
        print("Prediction made")
        pred_label = np.argmax(preds, axis=1)[0]

        return jsonify({
            "prediction": class_names[pred_label],
            "probabilities": preds.tolist()
        })
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
