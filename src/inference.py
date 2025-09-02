import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import DiabetesNet # Required import for loading the full model

# 1. ⚙️ Load the trained model
# FIXED: Added weights_only=False to allow loading the saved model object
model = torch.load("./diabetes_model_full.pt", weights_only=False, map_location=torch.device('cpu'))
model.eval()

# 2. 📋 Define your new sample data
sample = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]] # A sample row for prediction

# 3. CORRECT SCALING PROCESS
# Load the original dataset to fit the scaler
df = pd.read_csv("../data/diabetes.csv")

# Apply the same cleaning as in training (replacing 0s)
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zeros:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# Isolate the features (X) to fit the scaler
X_train_full = df.drop(columns=["Outcome"]).values

# Fit the scaler on the full training dataset
scaler = StandardScaler()
scaler.fit(X_train_full)

# Transform the new sample using the properly fitted scaler
sample_scaled = scaler.transform(sample)

# 4. 🧠 Perform Inference
# Convert the scaled sample to a PyTorch tensor
input_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

# Make a prediction
with torch.no_grad():
    prediction = model(input_tensor)
    # Apply sigmoid to get probability and extract the value
    probability = torch.sigmoid(prediction).item()

print(f"🔮 Probability of having diabetes: {probability:.4f}")

# Optional: Print the binary prediction (0 or 1)
if probability > 0.5:
    print("✅ Prediction: Diabetes (1)")
else:
    print("❌ Prediction: No Diabetes (0)")