# Particle Simulator

This project is an extension of the JMU CS470 Final Project of Lauren Hartley, Jack Ball, Ye Hun (Samuel) Joo, and Josh Kuesters, from the Spring 2023 semester. Their original repository can be found here: https://github.com/ball2jh/particle-simulator.git

## Use Instructions

This project can be run either in parallel or in serial. The code should first be compiled using make, inside of the particle_simulator directory.

```
cd particle_simulator
make
```

Once the project has compiled, either `./app` or `./app_serial` will launch their respective versions of the program, either in parallel or serial. A list of additional command-line arguments can be found through the use of -h.

```
./app -h
    Usage: ./app [-n num_particles] [-s particle_size] [-e explode_from_center (OPTIONAL)] [-w sweep_and_prune | -t quad_tree | -g spatial_hash (OPTIONAL)]
```

Alternatively, `bash benchmark.sh` will run all of the benchmark scripts inside of `particle-simulator` and compile their results in `particle-simulator/test_results`.

## Parallelized

### Sweep and Prune

This algorithm was able to be partially parallelized through the use of a pair data structure, linking the indices of two particles that overlapped on the x-axis and sending all such overlapping pairs off to the GPU to resolve. However, we could not parallelize the sorting and pruning phase of the algorithm itself, as insertion sort (which was chosen for its low space-complexity and near-O(n) runtime with mostly sorted lists) cannot be parallelized. Similarly, the pruning phase could not be parallelized effectively using CUDA, as it is impossible to know which of the particles might be overlapping before the sweep, and the traversal relies on knowing that particles are only overlapping if their right-most edge has not been reached.

### Spatial Hashing

The spatial hashing algorithm was able to be parallelized using CUDA due to the independence of operations on particles in cells which allows multiple GPU threads to operate on different cells simultaneously without interfering with each other. This seemed to perform really well with large number of particles. One issue that was observed was when the particle size was large the collision detection didn't work very well. When particles would overlap they would either sometimes bounce off each other or jaggedly pass through each other. We believe this was due to either the hash function not distributing particles across the hash table correctly or the grid cells in the spatial hash were too large relative to the particle size, so multiple particles can exist in the same cell without necessarily being close enough to interact correctly.

## Serialized

### Quadtrees

This algorithm utilizes vectors rather than a large array to resemble a quadtree structure. It promotes flexibility in order to avoid wasting data for allocating space for unevenly dispersed particles. Therefore, the four children of the quadtree will only allocate as much space as necessary to store the particles in their respective areas. The primary advantage of this data type is the improvement to detecting collisions; it only searches a quadrant for potential collisions per particle rather than iterating over every particle per particle.

### Sweep and Prune

A basic sweep-and-prune algorithm has been implemented with the intention of cutting down on spurious comparisons between particles that have no good likelihood of colliding. This algorithm performs a simple insertion sort on a list of "edges" for the axis-aligned bounding boxes of each particle, and then "sweeps" a line across the list of edges to perform finer-grained collision detection only for objects which are touched by the line at the same time.

### Spatial Hashing

This spatial hashing algorithm has been implemented to optimize collision checks involving multiple particles by focusing checks only between particles that are in close proximity based on their locations in a hash grid. Each cell in the grid corresponds to a specific spatial location and can contain multiple particles. The algorithm uses a hash function to calculate each particle's grid coordinate and inserts the particle into the corresponding cell, it then retieves each particle in it's cell and the surrounding cells, and then checks if the coordinates of cells overlap causing a collision to be detected.
