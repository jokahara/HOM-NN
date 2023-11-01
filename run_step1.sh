#!/bin/bash
cd Step1
for i in $(seq 0 7);
do
    sbatch ../gpurun.sh ../hommani/cross_validate.py -i $i -f input.txt --init ../pretrained_model/best-$i.pt;
done
cd ..
