#!/bin/bash
DATA_H=../data.h
VW=${1:-vw}

echo > $DATA_H

$VW --quiet -d regression_data_1.txt -f regression_data_1.model -c -k --passes 100 --holdout_off
$VW --quiet -d regression_data_1.txt -i regression_data_1.model -t -p regression_data_1.pred

xxd -i regression_data_1.model >> $DATA_H
xxd -i regression_data_1.pred  >> $DATA_H

$VW --quiet -d regression_data_1.txt -f regression_data_no_constant.model -c -k --passes 100 --holdout_off --noconstant
$VW --quiet -d regression_data_1.txt -i regression_data_no_constant.model -t -p regression_data_no_constant.pred

xxd -i regression_data_no_constant.model >> $DATA_H
xxd -i regression_data_no_constant.pred  >> $DATA_H

$VW --quiet -d regression_data_2.txt -f regression_data_ignore_linear.model -c -k --passes 100 --holdout_off --ignore_linear a --ignore_linear b
$VW --quiet -d regression_data_2.txt -i regression_data_ignore_linear.model -t -p regression_data_ignore_linear.pred

xxd -i regression_data_ignore_linear.model >> $DATA_H
xxd -i regression_data_ignore_linear.pred  >> $DATA_H

$VW --quiet -d regression_data_2.txt -f regression_data_2.model -c -k --passes 100 --holdout_off
$VW --quiet -d regression_data_2.txt -i regression_data_2.model -t -p regression_data_2.pred

xxd -i regression_data_2.model >> $DATA_H
xxd -i regression_data_2.pred  >> $DATA_H

# testing interactions order 2
$VW --quiet -d regression_data_3.txt -f regression_data_3.model -c -k --passes 100 --holdout_off -q ab
$VW --quiet -d regression_data_3.txt -i regression_data_3.model -t -p regression_data_3.pred

xxd -i regression_data_3.model >> $DATA_H
xxd -i regression_data_3.pred  >> $DATA_H

# testing interactions order 3
$VW --quiet -d regression_data_4.txt -f regression_data_4.model -c -k --passes 2 --holdout_off --interactions abc
$VW --quiet -d regression_data_4.txt -i regression_data_4.model -t -p regression_data_4.pred

xxd -i regression_data_4.model >> $DATA_H
xxd -i regression_data_4.pred  >> $DATA_H

# testing interactions order 4
$VW --quiet -d regression_data_4.txt -f regression_data_5.model -c -k --passes 2 --holdout_off --interactions abcd
$VW --quiet -d regression_data_4.txt -i regression_data_5.model -t -p regression_data_5.pred

xxd -i regression_data_5.model >> $DATA_H
xxd -i regression_data_5.pred  >> $DATA_H

# testing large models
$VW --quiet -d regression_data_3.txt -f regression_data_6.model -c -k --passes 2 --holdout_off -q ab -b 33 --sparse_weights
$VW --quiet -d regression_data_3.txt -i regression_data_6.model -t -p regression_data_6.pred --sparse_weights

xxd -i regression_data_6.model >> $DATA_H
xxd -i regression_data_6.pred  >> $DATA_H

# testing feature hashing
$VW --quiet -d regression_data_7.txt -f regression_data_7.model -c -k --passes 2 --holdout_off
$VW --quiet -d regression_data_7.txt -i regression_data_7.model -t -p regression_data_7.pred

xxd -i regression_data_7.model >> $DATA_H
xxd -i regression_data_7.pred  >> $DATA_H

# multi-class classification
$VW --quiet -d multiclass_data_4.txt --csoaa_ldf m --csoaa_rank -q ab -k -c --holdout_off --passes 100 -f multiclass_data_4.model
$VW --quiet -d multiclass_data_4.txt -i multiclass_data_4.model -t -p multiclass_data_4.pred
# manual creation of (sort by action id) multiclass_data_4.pred.manual

xxd -i multiclass_data_4.model >> $DATA_H
xxd -i multiclass_data_4.pred >> $DATA_H

# epsilon greedy
$VW --quiet -d cb_data_5.txt --cb_explore_adf --epsilon 0.3 -q ab -k -c --holdout_off --passes 100 -f cb_data_5.model
$VW --quiet -d cb_data_5.txt -i cb_data_5.model -p cb_data_5.pred

xxd -i cb_data_5.model >> $DATA_H
xxd -i cb_data_5.pred >> $DATA_H

# softmax
$VW --quiet -d cb_data_5.txt --cb_explore_adf --softmax --lambda -2 -q ab -k -c --holdout_off --passes 5 -f cb_data_6.model
$VW --quiet -d cb_data_5.txt -i cb_data_6.model -p cb_data_6.pred

xxd -i cb_data_6.model >> $DATA_H
xxd -i cb_data_6.pred >> $DATA_H

# bag
$VW --quiet -d cb_data_5.txt --cb_explore_adf --bag 3 -q ab -k -c --holdout_off --passes 100 -f cb_data_7.model
$VW --quiet -d cb_data_5.txt -i cb_data_7.model -p cb_data_7.pred

xxd -i cb_data_7.model >> $DATA_H
xxd -i cb_data_7.pred >> $DATA_H

# bag + epsilon greedy
$VW --quiet -d cb_data_5.txt --cb_explore_adf --bag 5 --epsilon 0.27 -q ab -k -c --holdout_off --passes 100 -f cb_data_8.model
$VW --quiet -d cb_data_5.txt -i cb_data_8.model -p cb_data_8.pred

xxd -i cb_data_8.model >> $DATA_H
xxd -i cb_data_8.pred >> $DATA_H

# epsilon greedy + dr
# TODO: results are confusing (for both dr & mtr) as they don't match ips in a simple example
$VW --quiet -d cb_data_5.txt --cb_explore_adf --epsilon 0.3 --cb_type dr -q ab -k -c --holdout_off --passes 1000 -f cb_data_9.model
$VW --quiet -d cb_data_5.txt -i cb_data_9.model -p cb_data_9.pred

xxd -i cb_data_9.model >> $DATA_H
xxd -i cb_data_9.pred >> $DATA_H


$VW --quiet -d cats.txt --cats 4 --min_value=185 --max_value=23959 --bandwidth 3000 -f cats.model
$VW --quiet -d cats.txt -i cats.model -p cats.pred -t

xxd -i cats.model >> $DATA_H
xxd -i cats.pred >> $DATA_H