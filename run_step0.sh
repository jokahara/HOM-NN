#!/bin/bash
cd Step0
# sets up the dataset.pkl
sbatch ../run-lightning-test.sh ../hommani/cross_validate.py -i 0 -f input.txt;
sleep 10
for i in $(seq 1 7);
do
    sbatch ../gpurun.sh ../hommani/cross_validate.py -i $i -f input.txt;
done
cd ..