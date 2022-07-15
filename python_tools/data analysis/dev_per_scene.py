import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import seaborn as sns
import matplotlib.pyplot as plt
from sympy import I

df1 = pd.read_pickle(Path("/media/wehak/WD Passport/dataset_test_1_1/raw/data.pkl"))
df2 = pd.read_pickle(Path("/media/wehak/WD Passport/dataset_test_2_1/raw/data.pkl"))

fig, axs = plt.subplots(3, 1)
fig.set_size_inches(18/2.54, 10/2.54)

def plot_axis(axis, i, vector, lim):
    axs[i].set_ylim(lim)
    axs[i].axhline(y=0.0, linestyle="--", color="magenta")
    axs[i].plot(
        df1.loc[df1["model"] == "d-handle", "frame_n"], 
        df1.loc[df1["model"] == "d-handle", f"{vector.lower()}_diff_{axis.lower()}"], 
        "r--"
        )
    axs[i].plot(
        df2.loc[df2["model"] == "d-handle", "frame_n"], 
        df2.loc[df2["model"] == "d-handle", f"{vector.lower()}_diff_{axis.lower()}"], 
        "b:"
        )

for i, axis in enumerate(["X", "Y", "Z"]):
    plot_axis(axis, i, "tvec", [-.1,.1])

# axs[0].set_ylabel("Distance [m]")

# axs[1].set_ylim([-10, 10])
# axs[1].axhline(y=0.0, linestyle="--", color="magenta")
# axs[1].plot(df.index, df["euler_diff_z"], color="blue")
# axs[1].plot(df.index, df["euler_diff_y"], color="green")
# axs[1].plot(df.index, df["euler_diff_x"], color="red")
# axs[1].set_ylabel("Rotation [degrees]")
# axs[1].set_xlabel("Time [frames]")

plt.tight_layout()
# plt.savefig(f"{figure_path}/{datetime.now().strftime('%m%d%Y_%H%M%S.png')}", dpi=300)
plt.show()