import pandas as pd    
import numpy as np          
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt             
import seaborn as sns   




df = pd.read_csv("../data/diabetes.csv")


# 🟡 اطلاعات کلی
print("🔹 شکل دیتا:", df.shape)
print("🔹 ستون‌ها:", df.columns.tolist())
print("🔹 چند نمونه اول:\n", df.head())

# 🔍 چک مقادیر گمشده
print("\n🔎 مقادیر گمشده:\n", df.isnull().sum())

# 🧪 خلاصه آماری
print("\n📈 خلاصه آماری:\n", df.describe())



# 🧹 اصلاح مقادیر غیر منطقی (مثلاً: فشار خون = 0)
cols_with_zeros = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for col in cols_with_zeros:
    df[col] = df[col].replace(0,np.nan)
    df[col].fillna(df[col].median(),inplace=True)
    
    

# ✅ نرمال‌سازی ویژگی‌ها
scaler = MinMaxScaler()
features = df.drop("Outcome",axis=1)
scaled_features = scaler.fit_transform(features)




# DataFrame جدید نرمال‌شده
df_scaled = pd.DataFrame(scaled_features,columns=features.columns)
df_scaled["Outcome"] = df["Outcome"]



# 💡 نمایش چند نمونه:
print("\n📊 دیتای نرمال‌شده:\n", df_scaled.head())