#!/bin/bash

make clean
make

for size in 0.025 0.01
do
    echo "Particle Size: ${PARTICLE_SIZE}"
    echo "Brute Force"
    for n_particles in 100 250 
    do
        echo "---${n_particles} Particles---"
        timeout 100 ./app_serial -n $n_particles -s $size
    done

    echo
    echo "Sweep and Prune"
    for n_particles in 100 250 
    do
        echo "---${n_particles} Particles---"
        timeout 100 ./app_serial -n $n_particles -s $size -w
    done

    echo
    echo "Spatial Hash"
    for n_particles in 100 250 
    do
        echo "---${n_particles} Particles---"
        timeout 100 ./app_serial -n $n_particles -s $size -g
    done

    echo
    echo "Quad Tree"
    for n_particles in 100 250
    do
        echo "---${n_particles} Particles---"
        timeout 100 ./app_serial -n $n_particles -s $size -t
    done
    echo
done

# 100 250 500 750 1000 1500 2000 3000