#!/bin/bash
cd Step1
for i in $(seq 0 7);
    sbatch ../gpurun.sh ../hommani/cross_validate.py -i $i -f input.txt;
done
cd ..