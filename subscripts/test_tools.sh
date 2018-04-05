#/bin/sh

# 各スクリプトのテストを実施する。

#エラー数カウンター
chk_err=0

#チェック関数
chkret()
{
	if [ $? != 0 ];then
		echo "ERROR"
		chk_err=`expr $chk_err + 1`
		return 0
	fi
}

#変数定義
test_data_dir="test_data"
out_dir="test_out"
#test_fn="test_data14y5"
test_fn="dirdata"
id="ID"
flg="Flag_correct"
m_test_fn="nba_data"
m_id="player"
m_flg="pos"

#ST GA param #20hで1800
#pop=300
#gen=15
#seed=1
pop=10
gen=5
seed=1

learn_m_lst=("rf" "logistic" "svm") #"xgb")

#処理はここから
mkdir -p $out_dir
mkdir -p ../tools_for_daiwa/${out_dir}
mkdir -p ../tools_for_cv_2/${out_dir}
mkdir -p ../tools_for_shift_2/${out_dir}
mkdir -p ../tools_for_cv_multi/${out_dir}
mkdir -p ../tools_for_shift_multi/${out_dir}

echo "CT:1"
data=${test_data_dir}/${test_fn}.csv
python evaluate_by_cv.py $data 0 $id $flg $out_dir 5 2 rf 1 0;chkret
python evaluate_by_cv.py $data 0 $id $flg $out_dir 5 2 logistic 1 0;chkret
python evaluate_by_cv.py $data 0 $id $flg $out_dir 5 2 svm 0 0;chkret
python evaluate_by_cv.py $data 0 $id $flg $out_dir 5 2 xgb 0 0;chkret

echo "CT:2"
for learn_m in ${learn_m_lst[@]}; do
	python output_thresh_confmat.py ${out_dir}/${test_fn}_${learn_m^^}prob_mean.csv $flg $out_dir cv;chkret
done

echo "CT:3"
data=${test_data_dir}/${test_fn}_train.csv
python run_ga.py $data 1 $flg $pop $gen $seed rf > ${out_dir}/test_rf2.log;chkret
python run_ga.py $data 0 $flg $pop $gen $seed rf > ${out_dir}/test_rf2.log;chkret
python run_ga.py $data $id $flg $pop $gen $seed lnsvm > ${out_dir}/test_lnsvm.log;chkret
for learn_m in ${learn_m_lst[@]}; do
	python run_ga.py $data $id $flg $pop $gen $seed ${learn_m} > ${out_dir}/test_${learn_m}.log; chkret
done

echo "CT:4"
python plot_ga.py ${out_dir}/test_rf.log $out_dir;chkret

echo "CT:5"
sh get_best_result.sh lnsvm ${out_dir}/test_lnsvm.log test;chkret
for learn_m in ${learn_m_lst[@]}; do
	sh get_best_result.sh ${learn_m} ${out_dir}/test_${learn_m}.log test;chkret
done

echo "CT:6"
data=${test_data_dir}/${test_fn}_train.csv
data2=${test_data_dir}/${test_fn}_test.csv
python gene_ga_data.py $data $data2 ${out_dir}/03_cols_rf_test.csv 0 $flg;chkret
for learn_m in ${learn_m_lst[@]}; do
	python gene_ga_data.py $data $data ${out_dir}/03_cols_${learn_m}_test.csv $id $flg;chkret
	python gene_ga_data.py $data $data2 ${out_dir}/03_cols_${learn_m}_test.csv $id $flg;chkret
done

echo "CT:7"
data=${test_data_dir}/${test_fn}.csv
data1=${test_data_dir}/${test_fn}_train.csv
data2=${test_data_dir}/${test_fn}_test.csv
str=${test_data_dir}/${test_fn}_train_ga_
str2=${test_data_dir}/${test_fn}_test_ga_
str3=${out_dir}/04_params_
for learn_m in ${learn_m_lst[@]}; do
	python evaluate.py $data1 $data2 0 $id $flg $out_dir ${learn_m};chkret
	python evaluate.py ${str}${learn_m}.csv ${str2}${learn_m}.csv ${str3}${learn_m}_test.csv $id $flg $out_dir ${learn_m};chkret
done

#カンマ区切りのテスト
python evaluate.py $data1,${str}rf.csv,$data $data2,${str2}rf.csv,$data 0,${str3}rf_test.csv,0 $id $flg $out_dir rf;chkret

echo "CT:8"
str=${out_dir}/${test_fn}_test_ga_
for learn_m in ${learn_m_lst[@]}; do
	python output_thresh_confmat.py ${str}${learn_m}_${learn_m^^}prob.csv $flg $out_dir shft;chkret
done

echo "CT:9"
str=${out_dir}/${test_fn}_test_ga_
for learn_m in ${learn_m_lst[@]}; do
	python plot_confmat.py ${str}${learn_m}_${learn_m^^}confusion_matrix.csv $out_dir;chkret
done

echo "CT:10"
str=${out_dir}/${test_fn}_test_ga_
for learn_m in ${learn_m_lst[@]}; do
	python plot_score_dist.py ${str}${learn_m}_${learn_m^^}prob.csv $out_dir shft;chkret
done

echo "CT:11-1"
data=${test_data_dir}/${test_fn}_train.csv
data2=${test_data_dir}/${test_fn}_test.csv
str2=${test_data_dir}/${test_fn}_train_ga_
str3=${out_dir}/04_params_
for learn_m in ${learn_m_lst[@]}; do
	python modeling.py $data 0 $id $flg $out_dir ${learn_m};chkret
	python modeling.py ${str2}${learn_m}.csv ${str3}${learn_m}_test.csv $id $flg $out_dir ${learn_m};chkret
done

echo "CT:11-2"
data=${test_data_dir}/${m_test_fn}.csv
for learn_m in ${learn_m_lst[@]}; do
	python modeling.py $data 0 $m_id $m_flg $out_dir ${learn_m};chkret
done

echo "CT:12-1"
data=${test_data_dir}/${test_fn}_test.csv
str=${out_dir}/${test_fn}_train_
str2=${out_dir}/${test_fn}_train_ga_
str3=${test_data_dir}/${test_fn}_test_ga_
for learn_m in ${learn_m_lst[@]}; do
	python predict.py $data ${str}${learn_m^^}_model.dump ${str}${learn_m^^}_cols.dump $id $flg $out_dir;chkret
	python predict.py ${str3}${learn_m}.csv ${str2}${learn_m}_${learn_m^^}_model.dump ${str2}${learn_m}_${learn_m^^}_cols.dump $id $flg $out_dir;chkret
done

echo "CT:12-2"
data=${test_data_dir}/${m_test_fn}.csv
str=${out_dir}/${m_test_fn}_
for learn_m in ${learn_m_lst[@]}; do
	python predict.py $data ${str}${learn_m^^}_model.dump ${str}${learn_m^^}_cols.dump $m_id $m_flg $out_dir;chkret
done

rm -f test_*.log

echo "ST:1"
cd ../tools_for_cv_2
data=../subscripts/${test_data_dir}/${test_fn}.csv
rm -f $out_dir/00*.log
for learn_m in ${learn_m_lst[@]}; do
	sh run_eval.sh $data $id $flg ${learn_m} $out_dir;chkret
	echo "sh run_ga_eval.sh $data $id $flg $pop $gen $seed ${learn_m} $out_dir"
	time(sh run_ga_eval.sh $data $id $flg $pop $gen $seed ${learn_m} $out_dir;chkret)
done

echo "ST:2"
cd ../tools_for_cv_multi
data=../subscripts/${test_data_dir}/${m_test_fn}.csv
rm -f $out_dir/00*.log
for learn_m in ${learn_m_lst[@]}; do
	if [ x${learn_m} == "xlogistic" ]; then
		continue
	fi
	sh run_eval.sh $data $m_id $m_flg ${learn_m} $out_dir;chkret
	echo "sh run_ga_eval.sh $data $m_id $m_flg $pop $gen $seed ${learn_m} $out_dir"
	time(sh run_ga_eval.sh $data $m_id $m_flg $pop $gen $seed ${learn_m} $out_dir;chkret)
done

echo "ST:3"
cd ../tools_for_shift_2
data1=../subscripts/${test_data_dir}/${test_fn}_train.csv
data2=../subscripts/${test_data_dir}/${test_fn}_test.csv
rm -f $out_dir/00*.log
for learn_m in ${learn_m_lst[@]}; do
	sh run_eval.sh $data1 $data2 $id $flg ${learn_m} $out_dir;chkret
	echo "sh run_ga_eval.sh $data1 $data2 $id $flg $pop $gen $seed ${learn_m} $out_dir"
	time(sh run_ga_eval.sh $data1 $data2 $id $flg $pop $gen $seed ${learn_m} $out_dir;chkret)
done

echo "ST:4"
cd ../tools_for_shift_multi
data=../subscripts/${test_data_dir}/${m_test_fn}.csv
rm -f $out_dir/00*.log
for learn_m in ${learn_m_lst[@]}; do
	if [ x${learn_m} == "xlogistic" ]; then
		continue
	fi
	sh run_eval.sh $data $data $m_id $m_flg ${learn_m} $out_dir;chkret
	echo "sh run_ga_eval.sh $data $data $m_id $m_flg $pop $gen $seed ${learn_m} $out_dir"
	time(sh run_ga_eval.sh $data $data $m_id $m_flg $pop $gen $seed ${learn_m} $out_dir;chkret)
done

echo "ST:5"
cd ../tools_for_daiwa
data=../subscripts/${test_data_dir}/${test_fn}.csv
sh run_eval.sh $data rf $out_dir;chkret
sh run_ga_eval.sh $data $pop $gen $seed rf $out_dir;chkret
#sh -x run_ga_eval.sh ../subscripts/${test_data_dir}/${test_fn}.csv 10 5 2 rf $out_dir;chkret
rm -f ${out_dir}/*.log
rm -f ${out_dir}/*.csv
rm -f ${out_dir}/rf_ga_list.dump
cd ../subscripts

cd ${out_dir}
rm -f test_*.log
rm -f test_rf_gaplot.png
rm -f 01_score_cols_params_*_test.csv
rm -f 02_score_*_test.csv
rm -f 03_cols_*_test.csv
rm -f 04_params_*_test.csv
rm -f ${test_fn}_imp.csv
rm -f ${test_fn}_ga_*.csv
rm -f ${test_fn}_train_*.csv
rm -f ${test_fn}_test_*.csv
rm -f ${test_fn}_*eva.csv
rm -f ${test_fn}_*imp.csv
rm -f ${test_fn}_*prob.csv
rm -f ${test_fn}_*eva_mean.csv
rm -f ${test_fn}_*imp_mean.csv
rm -f ${test_fn}_*prob_mean.csv
rm -f rf_ga_list.dump
rm -f evamean_summary.csv

rm -f 00_all_result_*_${test_fn}_train.log
rm -f 01_score_cols_params_*_${test_fn}_train.csv
rm -f 02_score_*_${test_fn}_train.csv
rm -f 03_cols_*_${test_fn}_train.csv
rm -f 04_params_*_${test_fn}_train.csv
cd ..

rm -f ${test_data_dir}/${test_fn}_imp.csv
rm -f ${test_data_dir}/${test_fn}_ga_*.csv
rm -f ${test_data_dir}/${test_fn}_train_*.csv
rm -f ${test_data_dir}/${test_fn}_test_*.csv
rm -f rf_ga_list.dump

exit $chk_err
