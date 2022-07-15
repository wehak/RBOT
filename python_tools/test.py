import numpy as np
from pathlib import Path
import csv
import pandas as pd
# x = np.random.normal(0, 1, 16).reshape([4,4])


x = np.random.normal(10, 4, 1000)
df = pd.DataFrame(x)
df["n"] = df.index

print(df[0].mean(), df[0].std())