import re
import numpy as np

# Read the log text
with open("Results.txt", "r") as f:
    log_data = f.read()

# Extract numerical values: [Epoch, Train Loss, Train AUC, Val Loss, Val AUC]
pattern = r"Epoch (\d+)/\d+ - Train Loss: ([\d.]+), AUC: ([\d.]+) - Val Loss: ([\d.]+), AUC: ([\d.]+)"
matches = re.findall(pattern, log_data)

# Convert strings to floats and create NumPy array
data_array = np.array(matches, dtype=float)

# Column mapping:
# data_array[:, 0] -> Epoch
# data_array[:, 1] -> Train Loss
# data_array[:, 2] -> Train AUC
# data_array[:, 3] -> Val Loss
# data_array[:, 4] -> Val AUC

print("Array Shape:", data_array.shape)  # Output: (30, 5)
print(data_array)

np.save("parsed_results.npy", data_array)