import numpy as np
import matplotlib.pyplot as plt
from dhflocalization.gridmap import GridMap

# distance transform plot
map_filename = "gt_map_01_table"
ogm = GridMap.load_map_from_config(map_filename)

fig, ax = plt.subplots()
fig.set_figwidth(7)
fig.set_figheight(5)
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

data = np.flipud(ogm.distance_transform)
# roi = [590:1350, 350:2100]

plt.imshow(data, cmap="gray", vmin=0, vmax=600)
plt.savefig("dt_map_whole.png")
plt.show()
