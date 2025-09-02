import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import df_scaled




# 🎯 جدا کردن ویژگی‌ها و برچسب‌ها
X = df_scaled.drop("Outcome", axis=1).values
y = df_scaled["Outcome"].values

# ✂️ تقسیم train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔁 تبدیل به Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 📦 ساخت Dataset و DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
