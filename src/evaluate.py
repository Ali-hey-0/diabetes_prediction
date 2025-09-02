
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Corrected the import path
from model import DiabetesNet 


df = pd.read_csv("../data/diabetes.csv")



X = df.drop(columns=["Outcome"]).values
y = df["Outcome"].values

# 🔢 نرمال‌سازی داده‌ها
scaler = StandardScaler()
X = scaler.fit_transform(X)

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Tensor کردن تست
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ⚙️ بارگذاری مدل
model = DiabetesNet(input_dim=X.shape[1])
# ❗️FIXED LINE: Added map_location to load model on a CPU-only machine
model.load_state_dict(torch.load("./checkpoints/diabetes_model.pth", map_location=torch.device("cpu")))
model.eval()

# 🎯 پیش‌بینی
with torch.no_grad():
    y_pred_probs = model(X_test)
    y_pred = (y_pred_probs >= 0.5).float()

# 📊 ارزیابی
accuracy = accuracy_score(y_test, y_pred.numpy())
conf_mat = confusion_matrix(y_test, y_pred.numpy())
report = classification_report(y_test, y_pred.numpy())

print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Confusion Matrix:\n", conf_mat)
print("\n📝 Classification Report:\n", report)
