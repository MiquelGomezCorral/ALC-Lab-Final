#!/bin/bash

python train.py --num_classes 2 --label_name task1 --emb_model qwen --not_phisio --annotators
python train.py --num_classes 3 --label_name task2 --emb_model qwen --balanced --not_phisio --annotators
python train.py --num_classes 6 --label_name task3 --emb_model qwen --multilabel --not_phisio --annotators