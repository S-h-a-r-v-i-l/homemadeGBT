# Example script demonstrating how to train and use the GBT model.
#
# Predicts house price from three features (age, sqft, interest_rate)
# on a small synthetic dataset, then reports accuracy and how error
# dropped over the course of boosting.
import numpy as np

from gbt.ensemble.booster import fit_ensemble
from gbt.ensemble.predictor import predict
from gbt.utils.metrics import rmse, mae, logcosh

np.random.seed(42)
num_rows = 100

age = np.random.uniform(0, 50, num_rows)               # years
sqft = np.random.uniform(600, 4000, num_rows)           # square feet
interest_rate = np.random.uniform(2.5, 7.5, num_rows)   # percent, at time of sale

price = (
    50_000
    + sqft * 150
    - age * 800
    - interest_rate * 8_000
    + np.random.normal(0, 15_000, num_rows)
)

features_array = np.array([age, sqft, interest_rate])

m = 150
predictor_inputs, bin_edges, base_prediction = fit_ensemble(features_array, price, m)

final_pred, all_predictions = predict(features_array, predictor_inputs, bin_edges, base_prediction)

print(f"Trained on {num_rows} houses with {m} trees.\n")

print("Sample predictions vs actual price:")
for i in range(5):
    print(
        f"  age={age[i]:5.1f}  sqft={sqft[i]:6.0f}  rate={interest_rate[i]:.2f}%  "
        f"-> predicted=${final_pred[i]:>10,.0f}   actual=${price[i]:>10,.0f}"
    )

print(f"\nRMSE:    ${rmse(price, final_pred):,.0f}")
print(f"MAE:     ${mae(price, final_pred):,.0f}")
print(f"Logcosh: {logcosh(price, final_pred):,.2f}")

print("\nConvergence (RMSE using only the first N trees, via staged predictions):")
for stage in (0, 24, 49, 99, m - 1):
    stage_rmse = rmse(price, all_predictions[stage])
    print(f"  after {stage + 1:3d} trees -> RMSE=${stage_rmse:,.0f}")
