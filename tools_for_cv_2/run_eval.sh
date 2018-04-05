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
# $1:データファイルを指定
# $2:ID識別キーのカラム名を指定 
# $3:目的変数のカラム名を指定。　正例1、負例0
# $4:学習器　rf,svm,logistic,xgb
# $5:出力先ディレクトリ
#
# NOTE:
#  プロセスをkillする場合は、呼び出し実行中の.pyもkillすること

####事前チェック###
if [ $# != 5 ];then
	echo "number of arguments is not correct"
	exit 1
fi

#引数取得
input_data=$1
index=$2
objective=$3
method=$4
out_dir=$5

in_fn=`basename $input_data .csv`

method_upper=`echo $method | tr "a-z" "A-Z"`


####出力先ディレクトリ作成####
mkdir -p $out_dir

#####MAIN########
####step4####
#評価
echo "step4"
python evaluate.py $input_data 0 $index $objective $out_dir 20 5 $method 0 0
if [ $? != 0 ];then
	echo "evaluate_by_cv.py ERROR"
	exit 1
fi

####step5####
#閾値を変えてフラグを付け直した時の評価値を出力
echo "step5"
python output_thresh_confmat.py $out_dir/${in_fn}_${method_upper}prob_mean.csv $objective $out_dir cv
if [ $? != 0 ];then
	echo "output_thresh_confmat.py ERROR"
	exit 1
fi

####step6####
#プロット
echo "step6"
python plot_confmat.py $out_dir/${in_fn}_${method_upper}confusion_matrix.csv $out_dir
if [ $? != 0 ];then
	echo "plot_confmat.py ERROR"
	exit 1
fi

python plot_score_dist.py $out_dir/${in_fn}_${method_upper}prob_mean.csv $out_dir cv
if [ $? != 0 ];then
        echo "plot_score_dist.py ERROR"
        exit 1
fi

exit 0
