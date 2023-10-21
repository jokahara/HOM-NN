#!/bin/bash
cd Step0
for i in $(seq 0 7);
do
    sbatch gpurun.sh ../hommani/cross_validate.py -i $i -f input.txt;
done
cd ..