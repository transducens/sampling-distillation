#!/bin/bash 

set -euo pipefail

if [ $# -lt 3 ]
then
  echo "Wrong number of arguments"
  exit 1
fi

lang1=$1
lang2=$2
permanentDir=$3
train_corpus=$4
dev_corpus=$5
test_corpus=$6
num_updates=$7

maxLegthAfterBpe=100

source train-steps-fairseq.sh
#########################################

prepare_data $train_corpus train
prepare_data $dev_corpus dev
prepare_data $test_corpus test

train_sentencepiece

sets="train dev test"
for set in $sets; do
  apply_sentencepiece $set $lang1 $permanentDir/spm/spm.$lang1.$lang2.model
  apply_sentencepiece $set $lang2 $permanentDir/spm/spm.$lang1.$lang2.model
done;

preprocess train
train_nmt

translate_test_spm train test

report train test

