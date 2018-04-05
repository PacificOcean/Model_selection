#!/bin/sh
# 
# GAの実行から評価までをまとめて実行 　CV式多値分類用
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
#
# ※以降は、テストデータ・学習データが同じで、組み合わせを変えてスコアリング
# [step4]
#  evaluate.pyで20回学習・テストを変えて評価を実施
#  recalcu_confmat_cv_multi.pyで、スコアの平均値から精度指標を再計算
#
# $1:データファイルを指定(学習兼テストデータ)
# $2:ID識別キーのカラム名を指定
# $3:目的変数のカラム名を指定
# $4:GAの探索で使用するpopulation
# $5:GAの探索で使用するgeneration
# $6:GAの探索で使用するseed
# $7:学習器　rf,svm,logistic,xgb
# $8:出力先ディレクトリ
#
# NOTE:
#  プロセスをkillする場合は、呼び出し実行中の.pyもkillすること

####事前チェック###
if [ $# != 8 ];then
	echo "number of arguments is not correct"
	exit 1
fi

#引数取得
input=$1
index=$2
objective=$3
ga_population=$4
ga_generation=$5
ga_seed=$6
method=$7
out_dir=$8

in_fn=`basename $input .csv`
input_path=`dirname $input`

method_upper=`echo $method | tr "a-z" "A-Z"`

#step1,2で使用
out_file00=${out_dir}/00_all_result_${method}_${in_fn}.log
#step3で使用
out_file03=${out_dir}/03_cols_${method}_${in_fn}.csv
#step4で使用
out_file04=${out_dir}/04_params_${method}_${in_fn}.csv

#ファイルの存在チェック。２重起動および、上書きを防止するため。
#途中から実行する場合はコメントアウトする。
if [ -f $out_file00 ];then
        echo "$out_file00 already exists."
        exit 1
fi

####出力先ディレクトリ作成####
mkdir -p $out_dir

#####MAIN########
####step1####
#GA実行
echo "step1"
python run_ga.py $input $index $objective $ga_population $ga_generation $ga_seed $method > $out_file00
if [ $? != 0 ];then
	echo "run_ga.py ERROR"
	exit 1
fi

####step2####
#ベストの結果を取得
echo "step2"
sh get_best_result.sh $method $out_file00 $in_fn
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
#GAで変数選択されたテストデータを出力。同じ場所にファイル名は「_ga_学習器」を付けて出力。
echo "step3"
python gene_ga_data.py $input $input $out_file03 $index $objective
if [ $? != 0 ];then
	echo "gene_ga_data.py ERROR"
	exit 1
fi

####step4####
#上記で出力したデータを使用して評価
echo "step4"
python evaluate.py ${input_path}/${in_fn}_ga_${method}.csv $out_file04 $index $objective $out_dir 20 5 $method 0 0
if [ $? != 0 ];then
	echo "evaluate_by_cv.py ERROR"
	exit 1
fi

#平均スコアを元にconfusionmatrixを再計算
python recalcu_confmat_cv_multi.py $out_dir/${in_fn}_ga_${method}_${method_upper}prob_mean.csv $objective $out_dir
if [ $? != 0 ];then
	echo "recalcu_confmat_cv_multi.py ERROR"
	exit 1
fi

exit 0
