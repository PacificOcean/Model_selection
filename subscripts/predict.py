# -*- coding: utf-8 -*-
#
# モデルのdumpファイル、予測対象データを与えて、予測を実施する。
# アウトプット：予測値、予測スコア
#
# argvs[1]:予測対象データ (カンマ区切りでリストも可)
# argvs[2]:モデルのdumpファイル (カンマ区切りでリストも可)
# argvs[3]:モデルの変数リストdumpファイル (カンマ区切りでリストも可)
# argvs[4]:ID識別キーのカラム名
# argvs[5]:目的変数のカラム名を指定 #無い場合は"0"を指定
# argvs[6]:出力先ディレクトリ

#----------基本モジュール----------
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import os
import sys
import math
import re #ワイルドカード等の正規表現で使用
import random
import shutil
import pickle

from sklearn import preprocessing


#####設定#####
argvs = sys.argv
in_fn = str(argvs[1])
model_dump = str(argvs[2])
model_col_dump = str(argvs[3])
index_col = str(argvs[4])
flag_col = str(argvs[5])
out_path = str(argvs[6])

index_col = index_col.decode('utf-8')
flag_col = flag_col.decode('utf-8')

#----------パス関連---------
Vec_Path_data=re.split(",", in_fn)
Vec_Path_model=re.split(",", model_dump)
Vec_Path_modelcol=re.split(",", model_col_dump)
#Vec_Path_data = [in_fn]#入力ファイルリスト

#出力先ディレクトリ
Path_rootdir_out = out_path +"/"

#----------データ関連---------
Sca_flagcol = flag_col #正解フラグの入っている変数名
Sca_flagkey = index_col #サンプルの識別キーが入っている変数名
Vec_exceptcol = [Sca_flagcol, Sca_flagkey] #推定に使わない変数名

Sca_ccode = "cp932" #出力の文字コード


#####予測実行##### 　各指標はCV毎の平均値とする 一時ディレクトリ使用しない版 パラレルなし版
loop_cnt=-1
for TMP_Path_data in Vec_Path_data:
    loop_cnt=loop_cnt+1
    #----------パス整理---------
    TMP_Path_data_split = re.split("\/", TMP_Path_data)
    TMP_filename_head = re.split("\.csv", TMP_Path_data_split[len(TMP_Path_data_split)-1])[0]
    TMP_Path_save = Path_rootdir_out

    #----------読み込み&データ準備---------
    print " start reading "+TMP_Path_data
    Mat_test = pd.read_csv(TMP_Path_data, encoding ='utf-8',dtype={Sca_flagkey:  object})
    print " end reading "+TMP_Path_data

    with open(Vec_Path_model[loop_cnt], 'r') as f:
        model = pickle.load(f)
    with open(Vec_Path_modelcol[loop_cnt], 'r') as f:
        Mat_train_columns = pickle.load(f)
    
    #----------モデル投入のためのデータ整理----------
    Mat_test_key = Mat_test[Sca_flagkey]
    TMP_Vec = np.setdiff1d(Mat_train_columns.values,np.asarray(Vec_exceptcol))
    Mat_test = Mat_test[TMP_Vec]

    #----------モデリング実行----------
    #正規化する場合、コメントイン ↓から
    std_scale = preprocessing.StandardScaler().fit(Mat_test)
    Mat_test = std_scale.transform(Mat_test)
    # ↑まで

    #フィッティング
    output = model.predict(Mat_test)

    #----------結果整理・保存----------
    ###予測値
    Mat_predict = DataFrame(np.c_[Mat_test_key.values, output], columns=[Sca_flagkey,"predict"])
    ###ラベル確率
    #TMP_Mat = np.c_[Mat_test_key.values, Vec_label_test.values]
    TMP_Mat_2 = model.predict_proba(Mat_test)
    if len(np.arange(TMP_Mat_2.shape[1]))==2: #2値の場合は、1となる確率のみを出力
        Mat_RFprob = DataFrame(np.c_[Mat_test_key.values, TMP_Mat_2[:,1]], columns=[Sca_flagkey,"prob_1"])
    else:
        Mat_RFprob = DataFrame(np.c_[Mat_test_key.values, TMP_Mat_2], columns=[Sca_flagkey]+["prob_" + str(i) for i in list(np.arange(TMP_Mat_2.shape[1]))])
    ###出力
    Mat_predict.to_csv(TMP_Path_save + TMP_filename_head +"_"+"predict.csv", index=False, encoding="utf-8")
    Mat_RFprob.to_csv(TMP_Path_save + TMP_filename_head +"_"+"prob.csv", index=False, encoding="utf-8")

