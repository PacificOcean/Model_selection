#!/bin/sh

# run_ga.pyが実行中でも、その時点のベストの結果を出力する。GAのログファイルと同じ場所に出力する。
#
# $1:学習器　rf,svm,logistic,xgb
# $2:GAのログファイルのパス
# $3:出力ファイルに含める文字列

if [ $# != 3 ];then
	echo "Usage: get_best_result.sh <model> <logfile> <outfile_key>"
	echo "   ex) get_best_result.sh rf ./test1.log test1"
	exit 1
fi

method=$1
ga_logfile=$2
outfile_key=$3

out_dir=`dirname $ga_logfile`

out_file00=${ga_logfile}
out_file01=${out_dir}/01_score_cols_params_${method}_${outfile_key}.csv
out_file02=${out_dir}/02_score_${method}_${outfile_key}.csv
out_file03=${out_dir}/03_cols_${method}_${outfile_key}.csv
out_file04=${out_dir}/04_params_${method}_${outfile_key}.csv

mkdir -p $out_dir

#ベストの結果を出力

#スコア+変数+パラメータ
grep -v ^gene $out_file00 | #gene行を除く
grep -v ^self.fit | #self.fit行を除く
tr '\n' 'X' | #改行コードをXに変換。★パラメタ等にXを含むものがある場合は他の文字にする必要がある★
sed -e "s/Xcols/ cols/g" | #colsの直前の改行コードを無くす
sed -e "s/Xparams/ params/g" | #paramsの直前の改行コードを無くす
tr 'X' '\n' | #改行コードを元に戻す
grep ^score | #scoreから始まる行を抽出する。cols、paramsも含まれている
sort -r -k 3 | #スコア値でソートする
head -n 1 | #スコアがベストである1行目を抽出
awk '{print $3;for(i=6;i<NF;i++){printf("%s",$i)}print $NF}' | #スコア値と変数番号以降を抽出
sed -e "s/\[//g" | #[を削除
sed -e "s/\]//g" | #]を削除
#sed -e "s/params:/\n/g" > $out_file01 ##params:を改行コードに変換して出力
sed -e "s/params:{/\n/g" | #params:{を改行コードに変換
sed -e "s/}//g" | #}を削除
sed -e "s/':/=/g" | sed -e "s/'//g" > $out_file01 #不要な記号を削除して出力

#スコアのみ
head -n 1 $out_file01 > $out_file02
#変数のみ
head -n 2 $out_file01 |tail -n 1 > $out_file03
#パラメタのみ
tail -n 1 $out_file01 > $out_file04

exit 0
