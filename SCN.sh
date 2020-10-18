#!/bin/bash

for loop in 8 7 6 5 4 3 2;do
#    python SCN_cnn.py --turn $loop | tee 'log/'$loop'_SCN.log'
    echo $loop"loop"
    python SCN_c.py --turn $loop
 #   echo "The value is:$loop"
done
