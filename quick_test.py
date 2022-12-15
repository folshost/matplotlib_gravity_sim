
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

NUM_POINTS = 5

metadata = np.random.random((5*NUM_POINTS))
metadata = metadata.reshape((5, -1))

# x positions
metadata[0, :] = metadata[0, :]*2000 - 1000
# y positions
metadata[1, :] = metadata[1, :]*2000 - 1000
# x velocity
metadata[2, :] = metadata[2, :]
# y velocity
metadata[3, :] = metadata[3, :]
# masses
metadata[4, :] = metadata[4, :]*1000


# Gravity calculation is:
'''

   f = G*(m1*m2)/(r^2)

'''


def get_gravity_component(check_idx, metadata):
    G = 6.67e-11
    positions = metadata[[0,1], :].T
    masses = metadata[2, :].T
    idx_loc = positions[check_idx]
    idx_mass = metadata[2, :]
    
    # dist = np.linalg.norm(positions - idx_loc, axis=1)
    denom_x = (positions[0] - idx_loc[0])**2
    denom_y = (positions[1] - idx_loc[1])**2
    denom_x[check_idx] = -1
    denom_y[check_idx] = -1
    a_x = G*(idx_mass)/denom_x
    a_y = G*(idx_mass)/denom_y
    return np.vstack((a_x, a_y))

fig, ax = plt.subplots()




ani = animation.FuncAnimation(fig, self.update, interval=5, 
                                          init_func=self.setup_plot, blit=True)

scat = ax.scatter()
for i in range(100):
    for j in range(metadata.shape[0]):
        vel_mod = get_gravity_component(j, metadata)
        metadata[2:4,:] = metadata[2:4,:] + vel_mod

    plt.scatter(metadata[0], metadata[1], s=metadata[4], alpha=0.5)
    plt.show()









