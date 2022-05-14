import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import seaborn as sns
import matplotlib.pyplot as plt

def log_reader(data_file):

    data_file = Path(data_file)
    if not data_file.is_file():
        print("No file found.")
        exit()

    figure_path = Path("/home/wehak/Dropbox/ACIT master/bilder/figures")

    T_FIDUCIAL = 2
    T_RBOT = 3

    with open(data_file) as f:
        csv_data = csv.reader(f, delimiter=";")

        t = []
        n_fiducials = []
        t_fiducial = []
        t_rbot = []

        for i, row in enumerate(list(csv_data)[5:]):
            t.append(           float( row[0].replace(",",".") ))
            n_fiducials.append( float( row[1].replace(",",".") ))
            t_fiducial.append(  np.array( row[T_FIDUCIAL][1:-1].replace(",",".").split(" "), dtype=np.float64 ).reshape([4, 4]))
            t_rbot.append(      np.array( row[T_RBOT][1:-1].replace(",",".").split(" "), dtype=np.float64 ).reshape([4, 4]))

    if len(t) == 0:
        print("No lines read.")
        exit()

    data = {}
    data["t"] = t
    data["n_fiducials"] = n_fiducials
    data["t_fiducial"] = t_fiducial
    data["t_rbot"] = t_rbot

    df = pd.DataFrame(data)
    df.index.name = "frame"
    # print(df)

    def t_to_euler(row):
        r = R.from_matrix(row[0:3, 0:3])
        return r.as_euler("zyx", degrees=True)

    def t_to_tvec(row):
        return row[0:3, 3]

    def set_nan_euler(row):
        if not row.t_fiducial.any():
            return [np.nan] * 3
        else:
            return row.euler_diff

    def set_nan_tvec(row):
        if not row.t_fiducial.any():
            return [np.nan] * 3
        else:
            return row.tvec_diff

    df["euler_fiducial"] = df.apply(lambda row: t_to_euler(row.t_fiducial), axis=1)
    df["tvec_fiducial"] = df.apply(lambda row: t_to_tvec(row.t_fiducial), axis=1)

    df["euler_rbot"] = df.apply(lambda row: t_to_euler(row.t_rbot), axis=1)
    df["tvec_rbot"] = df.apply(lambda row: t_to_tvec(row.t_rbot), axis=1)
    df["tvec_rbot"] /= 1000

    df["euler_diff"] = df["euler_fiducial"] - df["euler_rbot"]
    df["tvec_diff"] = df["tvec_fiducial"] - df["tvec_rbot"]

    # remove blank values
    df["euler_diff"] = df.apply(lambda row: set_nan_euler(row), axis=1)
    df["tvec_diff"] = df.apply(lambda row: set_nan_tvec(row), axis=1)

    df[["euler_diff_z", "euler_diff_y", "euler_diff_x"]] = pd.DataFrame(df.euler_diff.tolist(), index=df.index)
    df[["tvec_diff_x", "tvec_diff_y", "tvec_diff_z"]] = pd.DataFrame(df.tvec_diff.tolist(), index=df.index)

    return df



data = Path("/media/wehak/Data/coco_master/dataset_test_1_1")

if not Path(f"{data}/data.pkl").is_file():
    first = True
    for x in data.glob("*/"):
        for y in x.glob("*.txt"):
            if first:
                first = False
                df = log_reader(y)
                df["model"] = x.name
            else:
                new_df = log_reader(y)
                new_df["model"] = x.name
                df = pd.concat([df, new_df])

    df.to_pickle(Path(f"{data}/data.pkl"))
else:
    df = pd.read_pickle(Path(f"{data}/data.pkl"))

    # fig, axs = plt.subplots(2, 1)
    # fig.set_size_inches(18/2.54, 10/2.54)
    # axs[0].set_ylim([-0.1, 0.1])
    # axs[0].axhline(y=0.0, linestyle="--", color="magenta")
    # axs[0].plot(df.index, df["tvec_diff_x"], color="red")
    # axs[0].plot(df.index, df["tvec_diff_y"], color="green")
    # axs[0].plot(df.index, df["tvec_diff_z"], color="blue")
    # axs[0].set_ylabel("Distance [m]")

    # axs[1].set_ylim([-10, 10])
    # axs[1].axhline(y=0.0, linestyle="--", color="magenta")
    # axs[1].plot(df.index, df["euler_diff_z"], color="blue")
    # axs[1].plot(df.index, df["euler_diff_y"], color="green")
    # axs[1].plot(df.index, df["euler_diff_x"], color="red")
    # axs[1].set_ylabel("Rotation [degrees]")
    # axs[1].set_xlabel("Time [frames]")

    # plt.tight_layout()
    # # plt.savefig(f"{figure_path}/{datetime.now().strftime('%m%d%Y_%H%M%S.png')}", dpi=300)
    # plt.show()