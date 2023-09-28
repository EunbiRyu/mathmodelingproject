import main as main
import numpy as np
from matplotlib import pyplot as plt
import pylab
import seaborn


def get_files(filename_tail):
    locations = np.load(
        'data/location' + filename_tail + '.npy')  # np.load: returns arrays stored in this file
    adjacency_mat = np.load('data/adjacency_mat' + filename_tail + '.npy')
    return locations, adjacency_mat


def run_simulation(locations, adjacency_mat):
    colony = main.AntColony(locations, adjacency_mat, 0, locations.shape[0] - 1,
                            timesteps=1000, decay=0.1, n_ants=100)
    colony.run()
    print(colony.best_path)
    print(colony.best_path_dist)

    return colony


def plot(colony, all_paths=False):
    plt.title("THE Best Path")

    if all_paths:
        for path in colony.all_finished_paths:
            plt.plot(colony.locations[:, 0][path],
                     colony.locations[:, 1][path])
            plt.scatter(colony.locations[:, 0][path],
                        colony.locations[:, 1][path])
        plt.title("All Paths")

    plt.plot(colony.locations[:, 0][colony.best_path],
             colony.locations[:, 1][colony.best_path])
    plt.scatter(colony.locations[:, 0][colony.best_path],
                colony.locations[:, 1][colony.best_path])

    plt.show()

def plot_pheromones(colony):
    p = np.copy(colony.pheromones)
    p[0][0] = 0
    p_norm = (p - np.min(p)) / (np.max(p) - np.min(p))
    seaborn.heatmap(p_norm)
    plt.show()

#    plot_pheromones(colony)

# filename_tail = '10x10_maze'
# locations, adj_mat = get_files(filename_tail)
# colony = run_simulation(locations, adj_mat)
# plot(colony, all_paths=True)









# # locations = np.load('data/location7x7.npy')
# # adjacency_mat = np.load('data/adjacency_mat7x7.npy')
#
# locations = np.load('data/location10x10_maze.npy')
# adjacency_mat = np.load('data/adjacency_mat10x10_maze.npy')
#
# # locations = np.load('data/location20x20_maze.npy')
# # adjacency_mat = np.load('data/adjacency_mat20x20_maze.npy')
#
# start_idx = 0
# end_idx = len(locations) - 1
#
# colony = main.AntColony(locations, adjacency_mat, start_idx, end_idx)
# colony.run()
# print(colony.best_path)
# print(colony.best_path_dist)
#
# # pylab.title("Best Path ($n = " + str(n) + "$ steps)")
# # plt.plot(colony.locations[colony.best_path])
# # pylab.savefig("ant_colony_best_path"+str(n)+".png",bbox_inches="tight",dpi=600)
# # pylab.show()
#
#
# plt.plot(colony.locations[:, 0][colony.best_path], colony.locations[:, 1][colony.best_path])
#
# for i in range(len(colony.best_path)):
#     plt.scatter(colony.locations[colony.best_path[i]][0], colony.locations[colony.best_path[i]][1])
# pylab.title("THE Best Path")
# plt.show()
#
# def plot_pheromones(colony):
#     p = np.copy(colony.pheromones)
#     p[0][0] = 0
#     p_norm = (p-np.min(p))/(np.max(p)-np.min(p))
#     seaborn.heatmap(p_norm)
#     plt.show()
#
#
# # plot pheromone matrix
# plot_pheromones(colony)
#
