#!/bin/bash

make clean

make

timeout 7 ./app -n 3 -s 0.2 -e