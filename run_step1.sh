#!/bin/bash
cd Step1
sbatch ../gpurun.sh active_learning.py -f input.txt;
cd ..