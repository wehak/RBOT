import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_pickle(Path("/media/wehak/WD Passport/dataset_test_1_1/raw/data.pkl"))
x_axis = df.loc[df["model"] == "d-handle", "frame_n"] / 30 

fig, axs = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(18/2.54, 10/2.54)
axs[0].set_xlim(60, 6*60)
axs[0].set_ylim([-0.2, 0.2])
axs[0].axhline(y=0.0, linestyle="--", color="magenta")
axs[0].plot(
    x_axis, 
    df.loc[df["model"] == "d-handle", "tvec_diff_x"], 
    "r--"
    )
axs[0].plot(
    x_axis, 
    df.loc[df["model"] == "d-handle", "tvec_diff_y"], 
    "g:"
    )
axs[0].plot(
    x_axis, 
    df.loc[df["model"] == "d-handle", "tvec_diff_z"], 
    "b-."
    )
axs[0].set_ylabel("Distance [m]")

axs[1].set_ylim([-15, 15])
axs[1].axhline(y=0.0, linestyle="--", color="magenta")
axs[1].plot(
    x_axis, 
    df.loc[df["model"] == "d-handle", "euler_diff_x"], 
    "r--"
    )
axs[1].plot(
    x_axis, 
    df.loc[df["model"] == "d-handle", "euler_diff_y"], 
    "g:"
    )
axs[1].plot(
    x_axis, 
    df.loc[df["model"] == "d-handle", "euler_diff_z"], 
    "b-."
    )
axs[1].set_ylabel("Rotation [degrees]")
axs[1].set_xlabel("Time [seconds]")

plt.tight_layout()
plt.savefig(f"/home/wehak/Dropbox/ACIT master/bilder/figures/{datetime.now().strftime('%m%d%Y_%H%M%S.png')}", dpi=300)
# plt.show()