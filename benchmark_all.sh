#!/bin/bash

cd particle-simulator
make clean
make
echo "//////////////// SERIAL TESTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
bash serial_benchmark.sh > test_outputs/serial.txt
echo
echo "//////////////// PARALLEL TESTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
bash parallel_benchmark.sh > test_outputs/parallel.txt
echo
echo "//////////////// SPATIAL HASH STRESS TESTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
bash spatial_hashing_benchmark.sh > test_outputs/spatial_hash_stress.txt