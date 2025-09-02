import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from model import DiabetesNet
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


device = "cuda" if torch.cuda.is_available else "cpu"


# 📥 داده‌ها
df = pd.read_csv("../data/diabetes.csv")

X = df.drop(columns=["Outcome"]).values
y = df["Outcome"].values

# 🔢 نرمال‌سازی داده‌ها
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🧪 تقسیم به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



# Convert To Tensor
X_train = torch.tensor(X_train,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)





# Building DataLoader
train_loader = DataLoader(TensorDataset(X_train,y_train),batch_size=32,shuffle=True)


# ⚙️ ساخت مدل
model = DiabetesNet(input_dim=X.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)




 # 🎯 آموزش مدل
for epoch in range(100):
    for batch_X,batch_y in train_loader:
        y_pred = model(batch_X)
        loss = criterion(y_pred,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
       # محاسبه وزن کلاس‌ها
        class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
        class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

        # تعریف loss با وزن
        criterion = nn.BCELoss(weight=torch.tensor([class_weights_dict[1]]).float().to(device))
        
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.4f}")
       
        
    torch.save(model.state_dict(), "checkpoints/diabetes_model.pth")
    print("✅ آموزش کامل شد و مدل ذخیره شد.")

# for batch_X,batch_y in train_loader:
#     writer = SummaryWriter()
# writer.add_graph(model, batch_X)
# writer.close()