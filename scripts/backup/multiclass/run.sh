#!/bin/bash

time=300
lr=0.005

for arch in 0 1 2 3 ; do

qsub gpu_run tfpython anke_3_class.py --arch $arch --time_limit $time --learning_rate $lr --train

qsub gpu_run tfpython anke_3_class.py --arch $arch --time_limit $time --learning_rate $lr --train --normal

done;


