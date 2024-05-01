#!/bin/bash

for size in 0.005 0.0025
do
    echo "Particle Size: ${size}"
    for n_particles in 1000 2500 5000 10000 20000
    do 
        echo "---${n_particles} Particles---"
        timeout 100 ./app -n $n_particles -s $size
    done
    echo
done