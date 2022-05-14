from pathlib import Path
import matplotlib.pyplot as plt

filepath = Path("output/log.txt")
data = filepath.read_text().split(",") # file string to list
data = [float(x) for x in data if x] # remove empty strings from list
fig, axs = plt.subplots(1, 1)
axs.plot(data)
plt.show()