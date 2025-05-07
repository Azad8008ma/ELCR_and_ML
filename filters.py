import pandas as pd
import numpy as np
import json
def save_dict_to_json(data, filename):
  with open(filename, 'w') as f:
    json.dump(data,f)

Age_intervals = {}
EF_intervals = {}
weight_intervals = {}
rice_use_intervals = {}
pasta_use_intervals = {}
ED_intervals = {}
RBA_intervals = {}

def filter_maker():

    df = pd.read_excel('./pro_ELCR_data2.xlsx')

    # Calculate intervals for each variable
    Age_Quarter = (np.max(df["Age"]) - np.min(df['Age'])) / 5
    weight_Quarter = (np.max(df["weight"]) - np.min(df['weight'])) / 5
    rice_use_Quarter = (np.max(df["rice use"]) - np.min(df['rice use'])) / 5
    pasta_use_Quarter = (np.max(df["pasta use"]) - np.min(df['pasta use'])) / 5
    EF_Quarter = (np.max(df["EF"]) - np.min(df['EF'])) / 5
    ED_Quarter = (np.max(df["ED"]) - np.min(df['ED'])) / 5
    RBA_Quarter = (np.max(df["As_RBA"] + df["Cr_RBA"] + df["Hg_RBA"] + df["Cd_RBA"] + df["Pb_RBA"]) - np.min(df["As_RBA"] + df["Cr_RBA"] + df["Hg_RBA"] + df["Cd_RBA"] + df["Pb_RBA"])) / 5

    

    for i in range(5):
        Age_intervals[f"Age_interval_{i+1}"] = [i * Age_Quarter + np.min(df['Age']), (i + 1) * Age_Quarter + np.min(df['Age'])]
        EF_intervals[f"EF_interval_{i+1}"] = [i * EF_Quarter + np.min(df['EF']), (i + 1) * EF_Quarter + np.min(df['EF'])]
        weight_intervals[f"weight_interval_{i+1}"] = [i * weight_Quarter + np.min(df['weight']), (i + 1) * weight_Quarter + np.min(df['weight'])]
        rice_use_intervals[f"rice_use_interval_{i+1}"] = [i * rice_use_Quarter + np.min(df['rice use']), (i + 1) * rice_use_Quarter + np.min(df['rice use'])]
        pasta_use_intervals[f"pasta_use_interval_{i+1}"] = [i * pasta_use_Quarter + np.min(df['pasta use']), (i + 1) * pasta_use_Quarter + np.min(df['pasta use'])]
        ED_intervals[f"ED_interval_{i+1}"] = [i * ED_Quarter + np.min(df['ED']), (i + 1) * ED_Quarter + np.min(df['ED'])]
        RBA_intervals[f"RBA_interval_{i+1}"] = [i * RBA_Quarter + np.min(df["As_RBA"] + df["Cr_RBA"] + df["Hg_RBA"] + df["Cd_RBA"] + df["Pb_RBA"]), (i + 1) * RBA_Quarter + np.min(df["As_RBA"] + df["Cr_RBA"] + df["Hg_RBA"] + df["Cd_RBA"] + df["Pb_RBA"])]
    

    return RBA_intervals, Age_intervals, ED_intervals, EF_intervals, pasta_use_intervals, rice_use_intervals, weight_intervals
filter_maker()
intervals={
   "RBAs": RBA_intervals,
   "Ages": Age_intervals,
   "EDs": ED_intervals,
   "EFs": EF_intervals,
   "pasta_use": pasta_use_intervals,
   "rice_use": rice_use_intervals,
   "weights": weight_intervals
}
save_dict_to_json(intervals, 'my_intervals.json')