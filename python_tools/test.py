import numpy as np
from pathlib import Path
import csv
# x = np.random.normal(0, 1, 16).reshape([4,4])


folderpath = Path("/media/wehak/WD Passport/dataset_test_1_1/raw/d-handle")

def log_reader(folderpath):
    data_file = list(folderpath.glob("measurement_log*.txt"))[0]
    if not data_file.is_file():
        print("No file found.")
        exit()

    T_RBOT = 3
    with open(data_file) as f:
        csv_data = csv.reader(f, delimiter=";")

        t_rbot = []
        for i, row in enumerate(list(csv_data)):
            t_rbot.append(      np.array( row[T_RBOT][1:-1].replace(",",".").split(" "), dtype=np.float64 ).reshape([4, 4]))

    return t_rbot
    


x = log_reader(folderpath)
print(x[50].tolist())

# x = list(folderpath.glob("measurement_log*.txt"))[0]
# print(x)