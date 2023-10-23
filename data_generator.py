import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import math

def recursive_division(level, div, no_points, domain, origin=(0,0)):
    """
    Generate a pointset, clustered at different spatial resolutions, based on recursive division

    Input:
    level, list containing the number of regions that receive points after each split (levels[i] is the number of cells at level [i])
    div, the number of splits along each axis for each level
    no_points, the total number of points to be placed
    domain, the side length of the domain to place the points in
    origin, the origin of the current cell
    """
    
    # If we're at the lowest level, generate a point set in the specificied cell
    if not level:
        x_points = np.random.randint(origin[0], origin[0]+domain, no_points)
        y_points = np.random.randint(origin[1], origin[1]+domain, no_points)
        return x_points, y_points
    
    # Else, find the next division cells to put points into
    cells = random.sample(range(0, div*div), level[0])
    x_cells = [(cell % div) for cell in cells]
    y_cells = [cell // div for cell in cells]
    
    # Divide no_points and domain into levels[0] roughly equal integers
    div_points = [no_points // level[0] + (1 if x < no_points % level[0] else 0)  for x in range (level[0])]
    div_domain = [domain // div + (1 if x < domain % div else 0) for x in range (div)]
    x_points = np.empty(0, dtype=int)
    y_points = np.empty(0, dtype=int)

    # For each cell that was selected to contain points, find their origin and recurse
    for i in range(0, len(cells)):
        origin_x = sum(div_domain[0:(x_cells[i])], origin[0]) if x_cells[i] != 0 else origin[0]
        origin_y = sum(div_domain[0:(y_cells[i])], origin[1]) if y_cells[i] != 0 else origin[1]

        tempx, tempy = recursive_division(level[1:], div, div_points[i], div_domain[i%3], (origin_x, origin_y))

        x_points = np.concatenate([x_points, tempx])
        y_points = np.concatenate([y_points, tempy])

    return x_points, y_points

def write_data(coords, name):
    """
    Write an input point set to a file

    Input:
    coords, the coordinates of the points
    name, the name of the file to write to
    """

    with open("data/" + name, "wb") as fp:
        pickle.dump(coords[0], fp)
        pickle.dump(coords[1], fp)

def read_data(name):
    """
    Read a point set from a file

    Input: 
    name, the name of the file in which the pointset is stored
    """
    
    with open(name, "rb") as fp:
        X = pickle.load(fp)
        Y = pickle.load(fp)
    return X, Y

def visualise(pointsets):
    """
    Visualise input point sets using matplotlib

    Input:
    pointsets, a list of point sets to visualise
    """

    fig, axes = plt.subplots(math.ceil(len(pointsets)/2), 2, sharex=True, sharey=True)
    fig.set_size_inches(4, 4)

    for i in range(0, len(pointsets)):
        axes[int(i/2)][i % 2].scatter(pointsets[i][0], pointsets[i][1],s=3,c="black")

    plt.show()

def main():
    uniform = recursive_division([25, 25], 5, 1000, 100)
    small = recursive_division([25, 2], 5, 1000, 100)
    med = recursive_division([5, 25], 5, 1000, 100)
    multi = recursive_division([5, 4], 5, 1000, 100)

    write_data(uniform, "uniform")
    write_data(small, "small")
    write_data(med, "med")
    write_data(multi, "multi")

    visualise([uniform, small, med, multi])

if __name__ == "__main__":
    main()