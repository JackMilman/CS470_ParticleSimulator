#!/bin/bash

for size in 0.01 0.005
do
    echo "Particle Size: ${size}"
    for n_particles in 100 250 500 750 1000 1500 2000
    do
        echo "---${n_particles} Particles---"
        echo "--Brute Force"
        timeout 100 ./app -n $n_particles -s $size
        echo "--Sweep and Prune--"
        timeout 100 ./app -n $n_particles -s $size -w
        echo "--Spatial Hash--"
        timeout 100 ./app -n $n_particles -s $size -g
        echo
    done
done