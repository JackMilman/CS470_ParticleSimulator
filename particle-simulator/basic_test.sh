#!/bin/bash

make clean

make

# timeout 7 ./app -n 3 -s 0.2 -e

timeout 20 ./app_serial -n 100 -s 0.01
timeout 20 ./app_serial -n 1000 -s 0.01
timeout 20 ./app_serial -n 2000 -s 0.01
timeout 20 ./app_serial -n 3000 -s 0.01
timeout 20 ./app_serial -n 4000 -s 0.01
timeout 20 ./app_serial -n 5000 -s 0.01
timeout 20 ./app_serial -n 10000 -s 0.01

timeout 20 ./app_serial -n 100 -s 0.01 -w
timeout 20 ./app_serial -n 1000 -s 0.01 -w
timeout 20 ./app_serial -n 2000 -s 0.01 -w
timeout 20 ./app_serial -n 3000 -s 0.01 -w
timeout 20 ./app_serial -n 4000 -s 0.01 -w
timeout 20 ./app_serial -n 5000 -s 0.01 -w
timeout 20 ./app_serial -n 10000 -s 0.01 -w

timeout 20 ./app_serial -n 100 -s 0.01 -g
timeout 20 ./app_serial -n 1000 -s 0.01 -g
timeout 20 ./app_serial -n 2000 -s 0.01 -g
timeout 20 ./app_serial -n 3000 -s 0.01 -g
timeout 20 ./app_serial -n 4000 -s 0.01 -g
timeout 20 ./app_serial -n 5000 -s 0.01 -g
timeout 20 ./app_serial -n 10000 -s 0.01 -g