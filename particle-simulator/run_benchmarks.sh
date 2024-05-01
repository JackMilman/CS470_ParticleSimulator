#!/bin/bash

make clean
make

echo "//////////////// SERIAL TESTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
bash serial_benchmark.sh
echo
echo "//////////////// PARALLEL TESTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
bash parallel_benchmark.sh