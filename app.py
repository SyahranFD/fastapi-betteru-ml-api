from fastapi import FastAPI, File, UploadFile, HTTPException
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

# Model untuk input data nutrisi
class NutritionRequest(BaseModel):
    gender: str
    age: int
    weight: float
    height: float
    activity_level: str
    goal: str

# Model untuk output rekomendasi nutrisi
class NutritionResponse(BaseModel):
    profile: dict
    daily_nutrition: dict

class NutritionCalculator:
    def __init__(self, gender, age, weight, height, activity_level, goal):
        """
        Inisialisasi kalkulator nutrisi dengan parameter individu

        Parameters:
        - gender: 'Laki Laki' atau 'Perempuan'
        - age: umur dalam tahun
        - weight: berat badan dalam kg
        - height: tinggi badan dalam cm
        - activity_level: tingkat aktivitas fisik
        - goal: tujuan nutrisi
        """
        self.gender = gender
        self.age = age
        self.weight = weight
        self.height = height
        self.activity_level = activity_level
        self.goal = goal

        # Konstanta aktivitas
        self.activity_factors = {
            'sangat jarang beraktivitas': 1.2,
            'jarang beraktivitas': 1.375,
            'cukup beraktivitas': 1.55,
            'sering beraktivitas': 1.725,
            'sangat sering beraktivitas': 1.9
        }

        # Goal calorie adjustments
        self.goal_calorie_adjustments = {
            'menurunkan berat badan ekstrim': 0.6,
            'menurunkan berat badan': 0.8,
            'mempertahankan berat badan': 1.0,
            'menaikkan berat badan': 1.15,
            'menaikkan berat badan ekstrim': 1.25
        }

        if self.gender not in ['laki laki', 'perempuan']:
            raise HTTPException(status_code=400, detail="Jenis kelamin tidak valid, contoh yang valid adalah 'laki laki' atau 'perempuan'")

        if self.activity_level not in self.activity_factors:
            raise HTTPException(status_code=400, detail="Tingkat aktivitas tidak valid, contoh yang valid adalah 'sangat jarang beraktivitas', 'jarang beraktivitas', 'cukup beraktivitas', 'sering beraktivitas', atau 'sangat sering beraktivitas'")

        if self.goal not in self.goal_calorie_adjustments:
            raise HTTPException(status_code=400, detail="Tujuan tidak valid, contoh yang valid adalah 'menurunkan berat badan ekstrim', 'menurunkan berat badan', 'mempertahankan berat badan', 'menaikkan berat badan', atau 'menaikkan berat badan ekstrim'")

    def calculate_bmr(self):
        """
        Menghitung Basal Metabolic Rate (BMR) menggunakan
        rumus Harris Benedict
        """
        if self.gender == 'laki laki':
            bmr = 66 + (13.7 * self.weight) + (5 * self.height) - (6.8 * self.age)
        else:
            bmr = 655 + (9.6 * self.weight) + (1.8 * self.height) - (4.7 * self.age)

        return bmr

    def calculate_total_calories(self):
        """
        Menghitung total kalori harian
        """
        bmr = self.calculate_bmr()
        activity_factor = self.activity_factors.get(self.activity_level)
        goal_adjustment = self.goal_calorie_adjustments.get(self.goal)

        return bmr * activity_factor * goal_adjustment

    def calculate_macronutrients(self):
        """
        Menghitung kebutuhan makronutrien
        """
        total_calories = self.calculate_total_calories()

        # Protein calculation (15% of total calories)
        protein = 0.15 * total_calories / 4

        # Fat calculation (15% of total calories)
        fat = 0.15 * total_calories / 9

        # Carb calculation (60% of total calories)
        carb = 0.60 * total_calories / 4

        return {
            'Total Kalori': int(total_calories),
            'Protein': int(protein),
            'Lemak': int(fat),
            'Karbohidrat': int(carb)
        }

    def get_nutrition_recommendation(self):
        """
        Mendapatkan rekomendasi nutrisi lengkap
        """
        macronutrients = self.calculate_macronutrients()

        return {
            'Profil': {
                'Jenis Kelamin': self.gender,
                'Umur': self.age,
                'Berat Badan': self.weight,
                'Tinggi Badan': self.height,
                'Tingkat Aktivitas': self.activity_level,
                'Tujuan': self.goal
            },
            'Kebutuhan Nutrisi Harian': macronutrients
        }

@app.post("/calculate-nutrition", response_model=NutritionResponse)
async def calculate_nutrition(data: NutritionRequest):
    calculator = NutritionCalculator(
        gender=data.gender.lower(),
        age=data.age,
        weight=data.weight,
        height=data.height,
        activity_level=data.activity_level.lower(),
        goal=data.goal.lower()
    )

    recommendation = calculator.get_nutrition_recommendation()

    return NutritionResponse(
        profile=recommendation['Profil'],
        daily_nutrition=recommendation['Kebutuhan Nutrisi Harian']
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
