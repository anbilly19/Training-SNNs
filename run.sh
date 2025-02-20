#!/bin/bash

pip install --quiet spikingjelly
python main_spiking.py --train True #--load_model True
# python main_spiking.py --load_model True --single_test True
# python main_spiking.py --load_model True --single_test True --visualize True