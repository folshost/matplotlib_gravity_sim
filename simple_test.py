
import matplotlib.pyplot as plt
import numpy as np



#plt.ion()


fig, ax = plt.subplots()
scat = ax.scatter(np.random.random(10), 
                  np.random.random(10), 
                  c=np.random.random(10), 
                  s=np.random.random(10)*1000, 
                  vmin=0, vmax=1, cmap="jet",edgecolor="k", alpha=0.5)

plt.show()



