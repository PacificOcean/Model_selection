#!/bin/sh
# 
# GAの実行から評価までをまとめて実行
#
# INPUT: csvファイル、カラム名あり、ID識別キーあり、目的変数あり、欠損値は埋めること
# OUTPUT: 精度指標値(Precision,Recall,F-measure,Accurasy)
#
# [step1]
#  run_ga.pyでGAを実行
# [step2]
#  get_best_result.shでベストの結果を取得
# [step3]
#  gene_ga_data.pyで最適な説明変数を抽出したデータを作成 
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
# $5:GAの探索で使用するpopulation
# $6:GAの探索で使用するgeneration
# $7:GAの探索で使用するseed
# $8:学習器　rf,svm,logistic,xgb
# $9:出力先ディレクトリ
#
# NOTE:
#  プロセスをkillする場合は、呼び出し実行中の.pyもkillすること

####事前チェック###
if [ $# != 9 ];then
	echo "number of arguments is not correct"
	exit 1
fi

#引数取得
train_data=$1
test_data=$2
index=$3
objective=$4
ga_population=$5
ga_generation=$6
ga_seed=$7
method=$8
out_dir=$9

train_data_path=`dirname $train_data`
train_fn=`basename $train_data .csv`
test_data_path=`dirname $test_data`
test_fn=`basename $test_data .csv`

method_upper=`echo $method | tr "a-z" "A-Z"`

#step1,2で使用
out_file00=${out_dir}/00_all_result_${method}_${train_fn}.log
#step3で使用
out_file03=${out_dir}/03_cols_${method}_${train_fn}.csv
#step4で使用
out_file04=${out_dir}/04_params_${method}_${train_fn}.csv

#ファイルの存在チェック。２重起動および、上書きを防止するため。
#途中から実行する場合はコメントアウトする。
if [ -f $out_file00 ];then
        echo "$out_file00 already exists."
#        exit 1
fi


####出力先ディレクトリ作成####
mkdir -p $out_dir

#####MAIN########
####step1####
#GA実行
echo "step1"
python run_ga.py $train_data $index $objective $ga_population $ga_generation $ga_seed $method > $out_file00
if [ $? != 0 ];then
	echo "run_ga.py ERROR"
	exit 1
fi

####step2####
#ベストの結果を取得
echo "step2"
sh get_best_result.sh $method $out_file00 $train_fn
if [ $? != 0 ];then
	echo "get_best_result.sh ERROR"
	exit 1
fi
#経過をプロット
python plot_ga.py $out_file00 $out_dir
if [ $? != 0 ];then
	echo "plot_ga.py ERROR"
	#exit 1
fi

####step3####
#GAで変数選択されたデータを出力。同じ場所にファイル名は「_ga_学習器」を付けて出力。
echo "step3"
#学習データ
python gene_ga_data.py $train_data $train_data $out_file03 $index $objective
if [ $? != 0 ];then
	echo "gene_ga_data.py ERROR"
	exit 1
fi
#テストデータ
python gene_ga_data.py $train_data $test_data $out_file03 $index $objective
if [ $? != 0 ];then
	echo "gene_ga_data.py ERROR"
	exit 1
fi

####step4####
#上記で出力したデータを使用して評価
echo "step4"
python evaluate.py ${train_data_path}/${train_fn}_ga_${method}.csv ${test_data_path}/${test_fn}_ga_${method}.csv $out_file04 $index $objective $out_dir $method
if [ $? != 0 ];then
	echo "evaluate.py ERROR"
	exit 1
fi

####step5####
#閾値を変えてフラグを付け直した時の評価値を出力
echo "step5"
python output_thresh_confmat.py $out_dir/${test_fn}_ga_${method}_${method_upper}prob.csv $objective $out_dir shft
if [ $? != 0 ];then
	echo "output_thresh_confmat.py ERROR"
	exit 1
fi

####step6####
#プロット
echo "step6"
python plot_confmat.py $out_dir/${test_fn}_ga_${method}_${method_upper}confusion_matrix.csv $out_dir
if [ $? != 0 ];then
	echo "plot_confmat.py ERROR"
	exit 1
fi

python plot_score_dist.py $out_dir/${test_fn}_ga_${method}_${method_upper}prob.csv $out_dir shft
if [ $? != 0 ];then
	echo "plot_score_dist.py ERROR"
	exit 1
fi

exit 0
