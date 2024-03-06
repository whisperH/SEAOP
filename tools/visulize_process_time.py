# model_name = ["SEAOP"]
# Hela_time = [141.28+0.65]
# HCC_P_time = [47.37+0.20]
# HCC_T_time = [48.43+0.21]
# LASDC_N_time = [90.62+0.22]
# LASDC_T_time = [82.51+0.21]
import joblib
import os
model = joblib.load(
    os.path.join(
        "./Result/Hela_boost_repeat100",
        "model_name-CBLOF-n_clusters-12-contamination-0.02-seed_list-5-dataset_list-HelaGroups.joblib")
)
print(model)