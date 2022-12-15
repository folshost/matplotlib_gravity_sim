import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import time

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, metadata):
        self.metadata = metadata
        self.stream = self.data_stream()

        self.colors = np.random.rand(self.metadata.shape[1])

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=20, 
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        acc = next(self.stream)

        self.metadata[2] += acc[0]
        self.metadata[3] += acc[1]

        self.metadata[0] += self.metadata[2]
        self.metadata[1] += self.metadata[3]

        print("Metadata: ", metadata.shape)

        #testx = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        #testy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        #sizes = np.array([1.0, 20.0, 300.0, 4000.0, 50000.0])

        #self.scat = self.ax.scatter(testx, testy, c=self.colors, s=sizes,
        #                            cmap="jet", edgecolor="k", alpha=0.25)
        
        self.scat = self.ax.scatter(self.metadata[0], self.metadata[1], c=self.colors, s=self.metadata[4]/10, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k", alpha=0.5)
        self.ax.axis([-2000, 2000, -2000, 2000])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def get_gravity_component(self, check_idx):
        G = 6.674e-11*1000000000
        #positions = self.metadata[[0,1], :].T
        #print(positions)
        masses = self.metadata[4, :].T
        idx_loc = self.metadata[[0,1], check_idx]
        
        x_row = self.metadata[0].copy()
        y_row = self.metadata[1].copy()
        x_row[check_idx] = -1
        y_row[check_idx] = -1

        #print(self.metadata, x_row, y_row, idx_loc)
        #print("-----------------------------")

        #time.sleep(1)

        distances = np.sqrt((x_row - idx_loc[0])**2+(y_row - idx_loc[1])**2)
        #print("Distances: ", distances)
        x_components = x_row / distances
        y_components = y_row / distances
        #print("X and y components: ", x_components, y_components)
        
        a_x = G*(masses)/(distances**2)*x_components*-1
        a_y = G*(masses)/(distances**2)*y_components*-1

        # We don't want the idx "affecting" itself
        a_x[check_idx] = 0.0
        a_y[check_idx] = 0.0

        #print("Location: X: {0}, Y: {1}".format(idx_loc[0], idx_loc[1]))
        #print("Accelerations: ", a_x, a_y)
        
        return np.vstack((a_x, a_y))

    def data_stream(self):
        while True:
            tmp_val = np.zeros((2, self.metadata.shape[1]))
            for j in range(self.metadata.shape[0]):
                tmp_val += self.get_gravity_component(j)
            
            #print("Done!", tmp_val)
            yield tmp_val

    def update(self, i):
        """Update the scatter plot."""
        self.metadata[0] += self.metadata[2]
        self.metadata[1] += self.metadata[3]

        vel_mod = next(self.stream)

        self.metadata[2] += vel_mod[0]
        self.metadata[3] += vel_mod[1]

        #print(self.metadata)
        # Set x and y data...
        vals = self.metadata[:2].T
        self.scat.set_offsets(vals)
        # Set sizes...
        #self.scat.set_sizes(self.metadata[4])
        # Set colors..
        #self.scat.set_array(self.colors)

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,




if __name__ == '__main__':
    NUM_POINTS = 50

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
    metadata[4, :] = metadata[4, :]*10000

    #metadata = np.ones((5,5)).reshape((5, -1))
    #metadata = np.asarray([[1,2,3,4,5],
    #                       [1,2,3,4,5],
    #                       [1,2,3,4,5],
    #                       [1,2,3,4,5],
    #                       [1,2,3,4,5]], dtype=np.float64)
    #print("First: ", metadata)


    a = AnimatedScatter(metadata)
    
    plt.show()

