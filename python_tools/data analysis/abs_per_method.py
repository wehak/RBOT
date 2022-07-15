import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import seaborn as sns
import matplotlib.pyplot as plt
from sympy import I

# df1 = pd.read_pickle(Path("/media/wehak/WD Passport/dataset_test_1_1/raw/data.pkl"))
df2 = pd.read_pickle(Path("/media/wehak/WD Passport/dataset_test_2_1/raw/data.pkl"))
x_axis = df2.loc[df2["model"] == "d-handle", "frame_n"] / 60

fig, axs = plt.subplots(6, 1, sharex=True)
fig.set_size_inches(18/2.54, 15/2.54)
n = 3
axs[0].set_xlim([60*n, 60*(n+1)])

def plot_axis(axis, i, vector, lim):
    print(lim)
    axs[i].axhline(y=0.0, linestyle="--", color="magenta")
    axs[i].plot(
        x_axis, 
        df2.loc[df2["model"] == "d-handle", f"{vector}_rbot"].apply(lambda x: x[i]), 
        "r--"
        )
    axs[i].plot(
        x_axis, 
        df2.loc[df2["model"] == "d-handle", f"{vector}_fiducial"].apply(lambda x: x[i]), 
        "b:"
        )
    axs[i].set_ylim(lim)


y_lim = [[-100,100],[-100,100],[100,150]]
for i, axis in enumerate(range(3)):
    plot_axis(axis, i, "tvec", y_lim[i])

for i, axis in enumerate(range(3)):
    plot_axis(axis, i, "euler", [-90,90])


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