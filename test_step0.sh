#!/bin/bash
cd Step0
for i in $(seq 0 0);
do
    python3 ../hommani/cross_validate.py -i $i -f test.txt;
done
cd ..