#!/bin/bash

python train.py --num_classes 2 --label_name task1 --annotators --emb_model gemma
python train.py --num_classes 2 --label_name task1 --annotators --emb_model ville
python train.py --num_classes 2 --label_name task1 --emb_model qwen
python train.py --num_classes 3 --label_name task2 --annotators --emb_model gemma --balanced
python train.py --num_classes 3 --label_name task2 --annotators --emb_model ville --balanced
python train.py --num_classes 3 --label_name task2 --emb_model qwen --balanced
python train.py --num_classes 6 --label_name task3 --annotators --emb_model gemma --multilabel
python train.py --num_classes 6 --label_name task3 --annotators --emb_model ville --multilabel
python train.py --num_classes 6 --label_name task3 --emb_model qwen --multilabel