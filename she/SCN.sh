#!/bin/bash
mkdir log
for loop in 8 7 6 5 4 3 2
do
    #mkdir log
    python SCN_c.py --turn $loop | tee 'log/'$loop'SCN.log'
 #   echo "The value is:$loop"
done
