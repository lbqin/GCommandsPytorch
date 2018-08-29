#!/bin/bash

#rm -rf ./data

if [ ! -d "./data" ]; then
    rm train_list_noVAD.txt dev_list_noVAD.txt train_dev_list_noVAD.txt
    find "/dataset" -name '*train*.pcm' > train_list_noVAD.txt
    find "/dataset" -name '*dev*.pcm'   > dev_list_noVAD.txt
    cat train_list_noVAD.txt dev_list_noVAD.txt > train_dev_list_noVAD.txt

    mkdir ./data
    mkdir ./data/train
    mkdir ./data/valid

    for line in `cat train_list_noVAD.txt`
    do
        echo $line
        out=`echo $line|awk '{split($0,a,"/");print a[3]}'`
        if [ ! -d "./data/train/${out}" ]; then
            mkdir ./data/train/${out}
        fi
        cp $line ./data/train/${out}
    done

    for line in `cat dev_list_noVAD.txt`
    do
        echo $line
        out=`echo $line|awk '{split($0,a,"/");print a[3]}'`
        if [ ! -d "./data/valid/${out}" ]; then
            mkdir ./data/valid/${out}
        fi
        cp $line ./data/valid/${out}
    done
fi
mkdir ./result
python run.py --train_path ./data/train/ --valid_path ./data/valid/ --test_path ./data/valid/ --arc VGG9 --epochs 100 --patience 20 --window_size 0.03 --window_stride 0.02 --batch_size 50
