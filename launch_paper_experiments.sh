#!/bin/bash
set -euo pipefail



src_lang=$1 # Source language
tgt_lang=$2 # Target language
model=$3 # Teacher model
monolingual=$4 # Source corpus path
dev_corpus=$5 # Development set
test_corpus=$6 # Evaluation set


mkdir -p teacher_outputs

methods="beam_search diverse_beam_search top_p top_k" # Decoding methods
translations=10 # Number of translations per source sentence
all_groups="10 5 3 1" # Number of translations in each subset
sub_groups="5 3 1"


# Launch only a specific experiment
# python3 translate-pipeline.py $model $monolingual $src_lang $tgt_lang $method $translations > train-$method.$tgt_lang
# ./repeat_sentences.sh $monolingual train-$method.$src_lang
# python3 skip_empty_lines.py train-$method.$src_lang --tgt train-$method.$tgt_lang
# rm train-$method.$src_lang; rm train-$method.$tgt_lang
# mv train-$method.$src_lang-clean train-$method.$src_lang; mv train-$method.$tgt_lang-clean train-$method.$src_lang
# ./train-student.sh $src_lang $student_dir train-$method $dev $test $num_updates

for method in $methods; do    
    # Translate the monolingual corpora
    ##python3 translate-pipeline.py $model $monolingual $src_lang $tgt_lang $method $translations > teacher_outputs/$src_lang-$tgt_lang-$method

    # Prepare training data
    corpora=student_$src_lang-$tgt_lang-$method
    ##mkdir $corpora
    ##./repeat_sentences.sh $monolingual $corpora/train_$translations.$src_lang $translations # Source corpus
    ##cp teacher_outputs/$src_lang-$tgt_lang-$method $corpora/train_$translations.$tgt_lang # Target corpus

    ##for group in $sub_groups; do
    ##    paste $corpora/train_$translations.$src_lang $corpora/train_$translations.$tgt_lang | python3 get_subgroups.py --groups $group --output-file $corpora/train_$group.$src_lang-$tgt_lang
    ##    cut -f1 $corpora/train_$group.$src_lang-$tgt_lang > $corpora/train_$group.$src_lang
    ##    cut -f2 $corpora/train_$group.$src_lang-$tgt_lang > $corpora/train_$group.$tgt_lang
    ##    rm $corpora/train_$group.$src_lang-$tgt_lang
    ##done

    for group in $all_groups; do
        ##python3 skip_empty_lines.py --src $corpora/train_$group.$src_lang --tgt $corpora/train_$group.$tgt_lang
        ##rm $corpora/train_$group.$src_lang
        ##rm $corpora/train_$group.$tgt_lang
        ##mv $corpora/train_$group.$src_lang-clean $corpora/train_$group.$src_lang
        ##mv $corpora/train_$group.$tgt_lang-clean $corpora/train_$group.$tgt_lang

        # Launch training
        mkdir -p $corpora/$method-$group
        ./train-student.sh $src_lang $tgt_lang $corpora/$method-$group $corpora/train_$group $dev_corpus $test_corpus $(($group*1000))

        # The training corpus can be found in the student model direcory
        rm $corpora/train_$group.$src_lang
        rm $corpora/train_$group.$tgt_lang
    done

done