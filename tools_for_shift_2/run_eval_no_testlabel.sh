#!/bin/sh
# 
# 精度の評価処理をまとめて実行(run_ga_eval.shの後半部分:GAなし)
#
# INPUT: csvファイル、カラム名あり、ID識別キーあり、目的変数あり、欠損値は埋めること
# OUTPUT: 精度指標値(Precision,Recall,F-measure,Accurasy)
#
# [step4]
#  evaluate.pyで評価を実施
# [step5]
#  output_thresh_confmat.pyで閾値を変えてフラグを付け直した時の評価値を出力
# [step6]
#  plot_confmat.pyで閾値を変えてフラグを付け直した時の評価値をプロット
#
# $1:学習用データファイルを指定
# $2:テスト用データファイルを指定
# $3:ID識別キーのカラム名を指定
# $4:目的変数のカラム名を指定。　正例1、負例0
# $5:学習器　rf,svm,logistic,xgb
# $6:出力先ディレクトリ
#
# NOTE:
#  プロセスをkillする場合は、呼び出し実行中の.pyもkillすること

####事前チェック###
if [ $# != 6 ];then
	echo "number of arguments is not correct"
	exit 1
fi

#引数取得
train_data=$1
test_data=$2
index=$3
objective=$4
method=$5
out_dir=$6

test_fn=`basename $test_data .csv`

method_upper=`echo $method | tr "a-z" "A-Z"`


####出力先ディレクトリ作成####
mkdir -p $out_dir

#####MAIN########
####step4####
#評価
echo "step4"
python evaluate_no_testlabel.py $train_data $test_data 0 $index $objective $out_dir $method
if [ $? != 0 ];then
	echo "evaluate_no_testlabel.py ERROR"
	exit 1
fi

exit 0

####step5####
#閾値を変えてフラグを付け直した時の評価値を出力
echo "step5"
python output_thresh_confmat.py $out_dir/${test_fn}_${method_upper}prob.csv $objective $out_dir shft
if [ $? != 0 ];then
	echo "output_thresh_confmat.py ERROR"
	exit 1
fi

####step6####
#プロット
echo "step6"
python plot_confmat.py $out_dir/${test_fn}_${method_upper}confusion_matrix.csv $out_dir
if [ $? != 0 ];then
	echo "plot_confmat.py ERROR"
	exit 1
fi

python plot_score_dist.py $out_dir/${test_fn}_${method_upper}prob.csv $out_dir shft
if [ $? != 0 ];then
        echo "plot_score_dist.py ERROR"
        exit 1
fi

exit 0
