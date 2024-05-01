#!/bin/bash

make clean
make

for size in 0.0075 0.005 0.0025
do
    echo "Particle Size: ${size}"
    for n_particles in 100 250 500 1000 2500
    do 
        echo "---${n_particles} Particles---"
        timeout 100 ./app_serial -n $n_particles -s $size -t
    done
    echo
done