#!/bin/bash
cd Step1
python3 ../hommani/cross_validate.py -i 0 -f test.txt --init ../pretrained_model/best-0.pt;
cd ..