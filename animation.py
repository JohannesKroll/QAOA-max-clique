from matplotlib import pyplot, animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as ps
import math
import qutip

base_directory = '/home/jockel/master_2/QAOA/'
max_layer = 2
seeds = [42, 69, 8549, 2]
graphs = 5

fig = pyplot.figure()
ax = Axes3D(fig, azim=-40, elev=30)
sphere = qutip.Bloch(axes=ax)

# for index in range(graphs):
index = 4
for seed in seeds:
        for layer in range(max_layer):
            data = ps.read_csv(base_directory + 'angles_graph_' + str(index) + '_seed_' + str(seed) + '_layers_' + str(
                layer + 1) + '.csv')


            def animate(i):
                sphere.clear()
                for l in range(layer + 1):
                    x = math.sin(math.fabs(data['beta' + str(l)][i])) * math.cos(math.fabs(data['gamma' + str(l)][i]))
                    y = math.cos(math.fabs(data['beta' + str(l)][i])) * math.cos(math.fabs(data['gamma' + str(l)][i]))
                    z = math.sin(math.fabs(data['gamma' + str(l)][i]))
                    vec = [x, y, z]
                    sphere.add_vectors(vec)
                sphere.make_sphere()
                return ax


            def init():
                sphere.vector_color = ['r', 'b', 'g', 'c', 'm', 'y']
                return ax


            ani = animation.FuncAnimation(fig, animate, np.arange(len(data)),
                                          init_func=init, blit=False, repeat=False)
            ani.save('bloch_sphere_graph_' + str(index) + '_seed_' + str(seed) + '_layers_' + str(layer + 1) + '.mp4',
                     fps=6)
