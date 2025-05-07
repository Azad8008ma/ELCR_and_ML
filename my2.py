import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error

# بارگذاری داده‌ها
filepath = './pro_ELCR_data2.xlsx'
df = pd.read_excel(filepath)

# لیست ویژگی‌های عددی برای تحلیل رابطه با CR
numeric_features = df.select_dtypes(include=[np.number]).columns.drop("cr")

# ذخیره نتایج
results = []

# محاسبه‌ی ضرایب، دقت و خطای مدل برای هر ویژگی
for feature in numeric_features:
    x = df[feature].dropna()
    y = df["cr"].loc[x.index]  # هم‌راستا کردن مقادیر CR

    # بررسی یکنواخت نبودن x (برای جلوگیری از خطای هم‌شکل بودن مقادیر)
    if x.nunique() > 1:
        slope, intercept, r_value, _, _ = linregress(x, y)
        y_pred = slope * x + intercept  # مقادیر پیش‌بینی‌شده

        # محاسبه‌ی خطاها
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r_value ** 2  # ضریب تعیین

        results.append([feature, slope, intercept, r_value, r2, mae, mse, rmse])

# تبدیل نتایج به DataFrame
results_df = pd.DataFrame(results, columns=["Feature", "Slope", "Intercept", "R-value", "R²", "MAE", "MSE", "RMSE"])

# نمایش نتایج
print(results_df)

# ذخیره در فایل اکسل
results_df.to_excel("linear_regression_results.xlsx", index=False)
print("نتایج ذخیره شد: linear_regression_results.xlsx")
