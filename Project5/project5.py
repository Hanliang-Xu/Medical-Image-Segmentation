import json

from Project5.fastMarching import *

# Assuming lSNode and fastMarching classes are defined as per previous discussion
# Load project5.json content
with open('project5.json', 'r') as file:
  data = json.load(file)
test_dmap_init = np.array(data['test_dmap_init'])
test_dmap_init_3d = test_dmap_init[:, :, np.newaxis]  # Add a new axis

# Initialize the fastMarching class and update the distance map
fm = fastMarching()
fm.update(test_dmap_init_3d)

# Compute the mean absolute error (MAE) between the updated distance map and the initial map
mae = np.mean(np.abs(fm.dmap - test_dmap_init_3d))

print(f"Mean Absolute Error (MAE): {mae}")
