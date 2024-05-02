#!/bin/bash

for size in 0.005 0.0025
do
    echo "///Particle Size: ${size}\\\\\\"
    for n_particles in 100 250 500 1000 2500 5000 7500 10000 15000 20000
    do 
        echo "---${n_particles} Particles---"
        timeout 100 ./app -n $n_particles -s $size -g
    done
    echo
done