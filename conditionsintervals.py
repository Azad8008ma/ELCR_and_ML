import pandas as pd
import json
import numpy as np
from sklearn.linear_model import LinearRegression

# خواندن داده‌ها
arenges_file = pd.read_csv('./permutations_with_repetition.csv')
df = pd.read_excel('./pro_ELCR_data2.xlsx')

with open('my_intervals.json', 'r') as f:
    intervals = json.load(f)

# تابع عمومی برای ساخت بازه‌ها
def build_intervals(target_key, exclude_key):
    condition_intervals = []
    for _, row in arenges_file.iterrows():
        condition_interval = []
        for col in arenges_file.columns:
            if col == exclude_key:
                continue
            interval_key = f"{col}_{int(row[col])}"
            category = next(k for k in intervals.keys() if interval_key in intervals[k])
            condition_interval.append(intervals[category][interval_key])
        condition_intervals.append(condition_interval)
    return condition_intervals

# ساخت بازه‌ها برای هر ویژگی (بدون ویژگی مورد نظر)
age_intervals = build_intervals("Age", "1")
weight_intervals = build_intervals("Weight", "1.1")
rice_use_intervals = build_intervals("Rice Use", "1.2")
pasta_use_intervals = build_intervals("Pasta Use", "1.3")
ED_intervals = build_intervals("ED", "1.4")
EF_intervals = build_intervals("EF", "1.5")
RBA_intervals = build_intervals("RBA", "1.6")

# تابع عمومی برای پیدا کردن شیب میانگین
def mean_slopes_finder(target_feature, condition_intervals):
    slops = []
    for i in range(10):
        mask = (
            (df["weight"].between(np.min(condition_intervals[i][0]), np.max(condition_intervals[i][0]))) &
            (df["rice use"].between(np.min(condition_intervals[i][1]), np.max(condition_intervals[i][1]))) &
            (df["pasta use"].between(np.min(condition_intervals[i][2]), np.max(condition_intervals[i][2]))) &
            (df["ED"].between(np.min(condition_intervals[i][3]), np.max(condition_intervals[i][3]))) &
            (df["As_RBA"] + df["Cr_RBA"] + df["Hg_RBA"] + df["Cd_RBA"] + df["Pb_RBA"]).between(
                np.min(condition_intervals[i][4]), np.max(condition_intervals[i][4])
            ) &
            (df["EF"].between(np.min(condition_intervals[i][5]), np.max(condition_intervals[i][5])))
        )
        
        filtered_x = df.loc[mask, target_feature].values.reshape(-1, 1)
        filtered_y = df.loc[mask, "cr"].values.reshape(-1, 1)
        
        if len(filtered_x) >= 2:
            model = LinearRegression()
            model.fit(filtered_x, filtered_y)
            slops.append(model.coef_[0])
    
    return np.mean(slops) if slops else None

# محاسبه شیب میانگین برای هر ویژگی
mean_age_slope = mean_slopes_finder("Age", age_intervals)
mean_weight_slope = mean_slopes_finder("weight", weight_intervals)
mean_rice_use_slope = mean_slopes_finder("rice use", rice_use_intervals)
mean_pasta_use_slope = mean_slopes_finder("pasta use", pasta_use_intervals)
mean_ED_slope = mean_slopes_finder("ED", ED_intervals)
mean_EF_slope = mean_slopes_finder("EF", EF_intervals)
mean_RBA_slope = mean_slopes_finder("As_RBA", RBA_intervals)  # یا مقدار کلی RBA
