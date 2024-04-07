#!/bin/bash

# make clean

make

PARTICLE_SIZE=0.005

echo "Serial Brute Force Tests"
echo "Particle Size: ${PARTICLE_SIZE}"
echo "---100 Particles---"
timeout 100 ./app_serial -n 100 -s $PARTICLE_SIZE
echo "---500 Particles---"
timeout 100 ./app_serial -n 500 -s $PARTICLE_SIZE
echo "---1000 Particles---"
timeout 100 ./app_serial -n 1000 -s $PARTICLE_SIZE
echo "---2000 Particles---"
timeout 100 ./app_serial -n 2000 -s $PARTICLE_SIZE
echo "---3000 Particles---"
timeout 100 ./app_serial -n 3000 -s $PARTICLE_SIZE
echo "---4000 Particles---"
# timeout 100 ./app_serial -n 4000 -s $PARTICLE_SIZE
echo


echo "Sweep and Prune Tests"
echo "Particle Size: ${PARTICLE_SIZE}"
echo "---100 Particles---"
timeout 100 ./app_serial -n 100 -s $PARTICLE_SIZE -w
echo "---500 Particles---"
timeout 100 ./app_serial -n 500 -s $PARTICLE_SIZE -w
echo "---1000 Particles---"
timeout 100 ./app_serial -n 1000 -s $PARTICLE_SIZE -w
echo "---2000 Particles---"
timeout 100 ./app_serial -n 2000 -s $PARTICLE_SIZE -w
echo "---3000 Particles---"
timeout 100 ./app_serial -n 3000 -s $PARTICLE_SIZE -w
echo "---4000 Particles---"
# timeout 100 ./app_serial -n 4000 -s 0.005 -w
echo

echo "Spatial Hashing Tests"
echo "Particle Size: ${PARTICLE_SIZE}"
echo "---100 Particles---"
timeout 100 ./app_serial -n 100 -s $PARTICLE_SIZE -g
echo "---500 Particles---"
timeout 100 ./app_serial -n 500 -s $PARTICLE_SIZE -g
echo "---1000 Particles---"
timeout 100 ./app_serial -n 1000 -s $PARTICLE_SIZE -g
echo "---2000 Particles---"
timeout 100 ./app_serial -n 2000 -s $PARTICLE_SIZE -g
echo "---3000 Particles---"
timeout 100 ./app_serial -n 3000 -s $PARTICLE_SIZE -g
echo "---4000 Particles---"
# timeout 100 ./app_serial -n 4000 -s $PARTICLE_SIZE -g
echo

echo "Parallel Brute Force Tests"
echo "Particle Size: ${PARTICLE_SIZE}"
echo "---100 Particles---"
timeout 100 ./app -n 100 -s $PARTICLE_SIZE
echo "---500 Particles---"
timeout 100 ./app -n 500 -s $PARTICLE_SIZE
echo "---1000 Particles---"
timeout 100 ./app -n 1000 -s $PARTICLE_SIZE
echo "---2000 Particles---"
timeout 100 ./app -n 2000 -s $PARTICLE_SIZE
echo "---3000 Particles---"
timeout 100 ./app -n 3000 -s $PARTICLE_SIZE
echo "---4000 Particles---"
# timeout 100 ./app -n 4000 -s $PARTICLE_SIZE -g
echo

echo
echo
PARTICLE_SIZE=0.0025

echo "Brute Force Tests"
echo "Particle Size: ${PARTICLE_SIZE}"
echo "---100 Particles---"
timeout 100 ./app_serial -n 100 -s $PARTICLE_SIZE
echo "---500 Particles---"
timeout 100 ./app_serial -n 500 -s $PARTICLE_SIZE
echo "---1000 Particles---"
timeout 100 ./app_serial -n 1000 -s $PARTICLE_SIZE
echo "---2000 Particles---"
timeout 100 ./app_serial -n 2000 -s $PARTICLE_SIZE
echo "---3000 Particles---"
timeout 100 ./app_serial -n 3000 -s $PARTICLE_SIZE
echo "---4000 Particles---"
# timeout 100 ./app_serial -n 4000 -s $PARTICLE_SIZE
echo

echo "Sweep and Prune Tests"
echo "Particle Size: ${PARTICLE_SIZE}"
echo "---100 Particles---"
timeout 100 ./app_serial -n 100 -s $PARTICLE_SIZE -w
echo "---500 Particles---"
timeout 100 ./app_serial -n 500 -s $PARTICLE_SIZE -w
echo "---1000 Particles---"
timeout 100 ./app_serial -n 1000 -s $PARTICLE_SIZE -w
echo "---2000 Particles---"
timeout 100 ./app_serial -n 2000 -s $PARTICLE_SIZE -w
echo "---3000 Particles---"
timeout 100 ./app_serial -n 3000 -s $PARTICLE_SIZE -w
echo "---4000 Particles---"
# timeout 100 ./app_serial -n 4000 -s 0.005 -w
echo

echo "Spatial Hashing Tests"
echo "Particle Size: ${PARTICLE_SIZE}"
echo "---100 Particles---"
timeout 100 ./app_serial -n 100 -s $PARTICLE_SIZE -g
echo "---500 Particles---"
timeout 100 ./app_serial -n 500 -s $PARTICLE_SIZE -g
echo "---1000 Particles---"
timeout 100 ./app_serial -n 1000 -s $PARTICLE_SIZE -g
echo "---2000 Particles---"
timeout 100 ./app_serial -n 2000 -s $PARTICLE_SIZE -g
echo "---3000 Particles---"
timeout 100 ./app_serial -n 3000 -s $PARTICLE_SIZE -g
echo "---4000 Particles---"
# timeout 100 ./app_serial -n 4000 -s $PARTICLE_SIZE -g
echo "---5000 Particles---"
echo