#!/bin/bash

sudo apt install unzip
sudo pip3 install --force-reinstall gdown

gdown https://drive.google.com/u/0/uc?id=1CPMBmRj-jBDO2ax4CxkBs9iczIFrs8VA
mkdir -p ./data/blurred/train
mkdir -p ./data/blurred/test
mkdir -p ./data/sharp/train
mkdir -p ./data/sharp/test
unzip -j ./blurred_sharp.zip blurred_sharp/blurred/* -d ./data/blurred/train
unzip -j ./blurred_sharp.zip blurred_sharp/sharp/* -d ./data/sharp/train

for i in `seq 1130 1151`; do mv ./data/blurred/train/$i.png ./data/blurred/test; done
for i in `seq 1130 1151`; do mv ./data/sharp/train/$i.png ./data/sharp/test; done

python3 ./datasets/combine_A_and_B.py --fold_A ./data/blurred --fold_B ./data/sharp --fold_AB ./data/combined
