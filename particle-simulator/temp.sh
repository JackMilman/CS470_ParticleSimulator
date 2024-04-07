PARTICLE_SIZE=0.005

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

PARTICLE_SIZE=0.0025

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