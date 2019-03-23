#!/bin/bash

TEST_BASE=/home/yen/data/gern/samples_test/dataset/
for i in {00000000..00059999};
do
	if [ ! -e $TEST_BASE/$i/manifest.json ];
	then
		echo "Missing TEST $i"
	fi
done

TRAIN_BASE=/home/yen/data/gern/samples_train/dataset
for i in {00000000..00599999};
do
	if [ ! -e $TRAIN_BASE/$i/manifest.json ];
	then
		echo "Missing TRAIN $i"
	fi
done