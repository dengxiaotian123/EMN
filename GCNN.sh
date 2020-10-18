#!/bin/bash

#for loop in 8;do
    #python SCN_cnn.py --turn $loop | tee 'log/'$loop'_SCN.log'
 #   echo $loop"loop"
 #   python uGCNN_V3.py --turn 8 | tee 'log/'$loop'_GCNN.log'
python uGCNN_V3.py --turn 8 | tee log/8_GCNN.log
python uGCNN_V3.py --turn 7 | tee log/7_GCNN.log
python uGCNN_V3.py --turn 6 | tee log/6_GCNN.log
python uGCNN_V3.py --turn 5 | tee log/5_GCNN.log
python uGCNN_V3.py --turn 4 | tee log/4_GCNN.log
python uGCNN_V3.py --turn 3 | tee log/3_GCNN.log

 #   echo "The value is:$loop"
#done
