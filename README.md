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
    Usage: ./app_serial [-n num_particles] [-sp particle_size] [-e explosion (OPTIONAL)] [-w with_sweep (OPTIONAL)] [-h help (OPTIONAL)]
```

## Parallelized
The physics calculations and collision detection remain as of yet largely unchanged, with progress on the sweep-and-prune algorithm's parallelized form incomplete.

## Serialized

### Quadtrees

### Sweep and Prune
A basic sweep-and-prune algorithm has been implemented with the intention of cutting down on spurious comparisons between particles that have no good likelihood of colliding. This algorithm performs a simple insertion sort on a list of "edges" for the axis-aligned bounding boxes of each particle, and then "sweeps" a line across the list of edges to perform finer-grained collision detection only for objects which are touched by the line at the same time.