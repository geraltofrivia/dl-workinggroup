"""This file checks if the crucial dependencies are available. Run this script.
And if it does not break, you're golden."""

import numpy as np
npt = np.random.randn(10, 10)
print("Numpy is installed and imported successfully")

import pandas as pd
df = pd.DataFrame(npt)
print("Pandas is installed and imported successfully")

import torch
tt = torch.tensor(npt)
print("PyTorch is installed and imported successfully")

from matplotlib import pyplot as plt
plt.plot(np.random.randn(10))
# plt.show()
print("Matplotlib is installed and imported successfully")

print("All dependencies are installed and imported successfully. You're good")