from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
import io

# Load the trained TensorFlow model
model_path = 'model-food-indonesia.h5'  # Pastikan file model Anda ada di sini
model = tf.keras.models.load_model(model_path)

# Define the class labels
labels = [
    "ayam goreng", "burger", "french fries", "gado gado",
    "ikan goreng", "mie goreng", "nasi goreng", "nasi padang",
    "pizza", "rawon", "rendang", "sate", "soto"
]

app = FastAPI()


class PredictionResponse(BaseModel):
    prediction: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        img = Image.open(io.BytesIO(await file.read()))

        # Resize the image to 150x150 to match the model's expected input size
        img = img.resize((150, 150))  # Ubah ukuran menjadi 150x150
        img_array = np.array(img) / 255.0  # Normalisasi ke [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch

        # Predict using the TensorFlow model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = labels[predicted_class]

        return PredictionResponse(prediction=predicted_label)

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
