# Particle Simulator

This project is an extension of the JMU CS470 Final Project of Lauren Hartley, Jack Ball, Ye Hun (Samuel) Joo, and Josh Kuesters, from the Spring 2023 semester. Their original repository can be found here: https://github.com/ball2jh/particle-simulator.git

## Parallelized
The parallelized code has been updated to display wire-frame spheres in a 3-dimensional space. The physics calculations and collision detection remain as of yet largely unchanged, simply being expanded into 3-dimensions.

## Serialized
The serialized code remains in 2-dimensions. We intend to update this so that it also displays 3-dimensional particles. A basic sweep-and-prune algorithm has been implemented with the intention of cutting down on spurious comparisons between particles that have no good likelihood of colliding. This algorithm performs a simple insertion sort on a list of "edges" for the axis-aligned bounding boxes of each particle, and then "sweeps" a line across the list of edges to perform finer-grained collision detection only for objects which are touched by the line at the same time.

We have not yet been able to determine why, but even though this algorithm *should* perform better than the brute-force comparison algorithm, it appears to have worse performance with larger input sizes. Our preliminary guess is that the overhead cost of our implementation of the "sweep" phase of the algorithm may be unreasonably large.