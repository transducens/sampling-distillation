#!/bin/bash 

set -euo pipefail

# Variables to be set in the file 
# lang1=$1
# lang2=$2
# permanentDir=$3
# bpeOperations=$4
# trainCorpus=$5
# devCorpus=$6
# testCorpus=$7

gpuId=0
temp=/tmp

updates=$num_updates
trainArgs="--arch transformer --share-all-embeddings  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --weight-decay 0  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 --lr-scheduler inverse_sqrt --warmup-updates 8000 --warmup-init-lr 1e-7 --lr 0.0007 --save-interval-updates $updates  --patience 6 --no-progress-bar --max-tokens 8000 --update-freq 2 --eval-bleu --eval-bleu-args '{\"beam\":5,\"max_len_a\":1.2,\"max_len_b\":10}' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --keep-best-checkpoints 1 --keep-interval-updates 1 --no-epoch-checkpoints"


prepare_data () {

  prefix=$1  # Prefix to corpus 
  tag=$2 #train / dev / test

  echo "prepare_data $prefix $tag ######################"

  if [ ! -e $prefix.$lang1 ]
  then
    echo "prepare_data: ERROR: File $prefix.$lang1 does not exist"
    exit 1
  fi
  
    if [ ! -e $prefix.$lang2 ]
  then
    echo "prepare_data: ERROR: File $prefix.$lang2 does not exist"
    exit 1
  fi
  
  mkdir -p $permanentDir/corpus
  cat $prefix.$lang1 > $permanentDir/corpus/$tag.$lang1 
  cat $prefix.$lang2 > $permanentDir/corpus/$tag.$lang2 
}

train_sentencepiece () {
  if [[ ! -f $permanentDir/spm/spm.$lang1.$lang2.model ]]; then
    mkdir -p $permanentDir/spm
    current_dir=$PWD
    cd $permanentDir/spm
    cat $current_dir/$permanentDir/corpus/train.$lang1 $current_dir/$permanentDir/corpus/train.$lang2 | shuf > train_spm
    spm_train --input=train_spm --model_prefix=spm.$lang1.$lang2 --vocab_size=10000 --character_coverage=1.0 --model_type=bpe
    rm train_spm
    cd $current_dir
  fi
}

apply_sentencepiece () {
  prefix=$1
  lang=$2

  sp_model=$3
  echo "sentence_piece $prefix $lang ######################"

  if [ ! -e $permanentDir/corpus/$prefix.$lang ]
  then
    echo "sentence_piece: ERROR: File $permanentDir/corpus/$prefix.$lang does not exist"
    exit 1
  fi

  if [[ ! -f $permanentDir/corpus/$prefix.spm.$lang ]]; then
    python3 spm_encode.py --model $sp_model --output_format=piece --inputs=$permanentDir/corpus/$prefix.$lang --outputs=$permanentDir/corpus/$prefix.spm.$lang
  fi
}

preprocess () {
  echo "make_data_for_training $@ ######################"
  echo "preprocess -s $lang1 -t $lang2"

  for tag in "$@"
  do
    if [ ! -e $permanentDir/corpus/$tag.spm.$lang1 ]
    then
      echo "make_data_for_training: ERROR: File $permanentDir/corpus/$tag.spm.$lang1 does not exist"
      exit 1
    fi

    if [ ! -e $permanentDir/corpus/$tag.spm.$lang2 ]
    then
      echo "make_data_for_training: ERROR: File $permanentDir/corpus/$tag.spm.$lang2 does not exist"
      exit 1
    fi
  done

  fairseq-preprocess -s $lang1 -t $lang2  --trainpref $permanentDir/corpus/train.spm \
                     --validpref $permanentDir/corpus/dev.spm \
                     --testpref $permanentDir/corpus/test.spm \
                     --destdir $permanentDir/model/data-bin-train --workers 16 --joined-dictionary
}

train_nmt () {
  echo "train_nmt ######################"
  
  if [ ! -d $permanentDir/model/data-bin-train ]
  then
    echo "train_nmt_fairseq: ERROR: Folder $permanentDir/model/data-bin-train does not exist"
    exit 1
  fi

  echo "Training args: $trainArgs"
  echo "See $permanentDir/model/train.log for details"  

  eval "CUDA_VISIBLE_DEVICES=0 fairseq-train --task translation_multi_simple_epoch $trainArgs --seed $RANDOM --save-dir $permanentDir/model/checkpoints $permanentDir/model/data-bin-train &> $permanentDir/model/train.log"

  mv $permanentDir/model/checkpoints/checkpoint_best.pt $permanentDir/model/checkpoints/train.checkpoint_best.pt
  rm -fr $permanentDir/model/checkpoints/checkpoint* 
}

translate_test_spm () {
  tag=$1
  echo "translate_test $tag $lang1 - $lang2 ######################"

  if [ ! -e $permanentDir/model/checkpoints/$tag.checkpoint_best.pt ]
  then
    echo "translate_test_fairseq: ERROR: File $permanentDir/model/checkpoints/$tag.checkpoint_best.pt does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/corpus/test.spm.$lang1 ]
  then
    echo "translate_test_fairseq: ERROR: File $permanentDir/corpus/test.spm.$lang1 does not exist"
    exit 1
  fi

  if [ ! -d $permanentDir/model/data-bin-$tag ]
  then
    echo "train_nmt_fairseq: ERROR: Folder $permanentDir/model/data-bin-$tag does not exist"
    exit 1
  fi

  mkdir -p $permanentDir/eval/

  CUDA_VISIBLE_DEVICES=0 fairseq-interactive  --input $permanentDir/corpus/test.spm.$lang1 --path $permanentDir/model/checkpoints/$tag.checkpoint_best.pt \
                                              $permanentDir/model/data-bin-$tag --remove-bpe 'sentencepiece' | grep '^H-' | cut -f 3 > $permanentDir/eval/test.output-$tag
}

report () {
  tag=$1
  echo "report $tag ######################"

  if [ ! -e $permanentDir/eval/test.output-$tag ]
  then
    echo "report: ERROR: File $permanentDir/eval/test.output-$tag does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/corpus/test.$lang2 ]
  then
    echo "report: ERROR: File $permanentDir/corpus/test.$lang2 does not exist"
    exit 1
  fi
  cat $permanentDir/eval/test.output-$tag | sacrebleu $permanentDir/corpus/test.$lang2 --width 3 -l $lang1-$lang2 --metrics bleu chrf  > $permanentDir/eval/report-$tag
}
