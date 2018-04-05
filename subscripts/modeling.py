# -*- coding: utf-8 -*-
#
# 学習データ、テストデータを与えて、評価を実施する。
# アウトプット：予測スコア、重要度、評価指標、モデルのdumpファイル
#
# argvs[1]:学習データ (カンマ区切りでリストも可)
# argvs[2]:テストデータ (カンマ区切りでリストも可)
# argvs[3]:GAの結果のパラメータデータ (カンマ区切りでリストも可) #GA無しの場合は"0"とする。
# argvs[4]:ID識別キーのカラム名
# argvs[5]:目的変数のカラム名を指定
# argvs[6]:出力先ディレクトリ
# argvs[7]:学習器 rf,svm,logistic,xgb

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

#----------統計モデル----------
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm
from sklearn import cross_validation
from sklearn import tree
from sklearn.externals.six import StringIO

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import ShuffleSplit
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

#----------自作モジュール-------
import utils

#####設定#####
argvs = sys.argv
in_fn = str(argvs[1])
param_fn = str(argvs[2])
index_col = str(argvs[3])
flag_col = str(argvs[4])
out_path = str(argvs[5])
learn_m = str(argvs[6])

index_col = index_col.decode('utf-8')
flag_col = flag_col.decode('utf-8')

#----------パス関連---------
Vec_Path_data=re.split(",", in_fn)
#Vec_Path_data = [in_fn]#入力ファイルリスト

#GAで選択されたパラメタのデータ
if param_fn != "0":
    Vec_Path_paramdata = re.split(",", param_fn)

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
    #print TMP_Path_data
    loop_cnt=loop_cnt+1
    #----------パス整理---------
    TMP_Path_data_split = re.split("\/", TMP_Path_data)
    TMP_train_filename_head = re.split("\.csv", TMP_Path_data_split[len(TMP_Path_data_split)-1])[0]
    TMP_Path_save = Path_rootdir_out

    #----------学習器パラメタ設定---------
    if learn_m=="rf":
        #GAなし
        if param_fn == "0":
            model = RandomForestClassifier(n_jobs=1, random_state=1)#, class_weight={0:1, 1:40})
        elif Vec_Path_paramdata[loop_cnt] == "0":
            model = RandomForestClassifier(n_jobs=-1, random_state=1)#, class_weight={0:1, 1:40})
        #GAあり
        else:
            #パラメタ取得
            params=np.loadtxt(Vec_Path_paramdata[loop_cnt], delimiter=',',dtype=object)
            para_n_jobs=utils.get_param(params,"n_jobs")
            para_min_samples_leaf=utils.get_param(params,"min_samples_leaf")
            para_n_estimators=utils.get_param(params,"n_estimators")
            para_min_samples_split=utils.get_param(params,"min_samples_split")
            para_random_state=utils.get_param(params,"random_state")
            para_max_features=utils.get_param(params,"max_features")
            para_max_depth=utils.get_param(params,"max_depth")

            #パラメタ反映
            model = RandomForestClassifier(n_jobs=int(para_n_jobs),
                                           #n_jobs=-1,
                                           min_samples_leaf=int(para_min_samples_leaf),
                                           n_estimators=int(para_n_estimators),
                                           min_samples_split=int(para_min_samples_split),
                                           random_state=int(para_random_state),
                                           max_features=float(para_max_features),
                                           max_depth=int(para_max_depth))
    elif learn_m=="logistic":
        #GAなし
        if param_fn == "0":
            model = LogisticRegression()
        elif Vec_Path_paramdata[loop_cnt] == "0":
            model = LogisticRegression()
        #GAあり
        else:
            #パラメタ取得
            params=np.loadtxt(Vec_Path_paramdata[loop_cnt], delimiter=',',dtype=object)
            para_C=utils.get_param(params,"C")
            para_random_state=utils.get_param(params,"random_state")
            #パラメタ反映
            model = LogisticRegression(C=float(para_C),
                        random_state=int(para_random_state))
    elif learn_m=="svm":
        #GAなし
        if param_fn == "0":
            model = SVC(C=100000,probability=True)
        elif Vec_Path_paramdata[loop_cnt] == "0":
            model = SVC(C=100000,probability=True)
        #GAあり
        else:
            #パラメタ取得
            params=np.loadtxt(Vec_Path_paramdata[loop_cnt], delimiter=',',dtype=object)
            para_C=utils.get_param(params,"C")
            #para_degree=utils.get_param(params,"degree")
            #para_gamma=utils.get_param(params,"gamma")
            #para_coef0=utils.get_param(params,"coef0")
            #para_tol=utils.get_param(params,"tol")
            para_random_state=utils.get_param(params,"random_state")
            #para_probability=utils.get_param(params,"probability")
            #パラメタ反映
            model = SVC(C=float(para_C),
                        #degree=int(para_degree),
                        #gamma=float(para_gamma),
                        #coef0=float(para_coef0),
                        #tol=float(para_tol),
                        random_state=int(para_random_state),
                        #probability=bool(para_probability))
                        probability=True)
    elif learn_m=="lnsvm": #.predict_probaはLinearSVC()に対しては使えないため、未サポート
        #model = LinearSVC()
        print "LinearSVC is not supported"
    elif learn_m=="xgb":
        #GAなし
        if param_fn == "0":
            model = xgb.XGBClassifier()
        elif Vec_Path_paramdata[loop_cnt] == "0":
            model = xgb.XGBClassifier()
        #GAあり
        else:
            #パラメタ取得
            params=np.loadtxt(Vec_Path_paramdata[loop_cnt], delimiter=',',dtype=object)
            para_gamma=utils.get_param(params,"gamma")
            para_max_depth=utils.get_param(params,"max_depth")
            para_min_child_weight=utils.get_param(params,"min_child_weight")
            para_max_delta_step=utils.get_param(params,"max_delta_step")
            para_subsample=utils.get_param(params,"subsample")
            #para_objective=utils.get_param(params,"objective")
            para_base_score=utils.get_param(params,"base_score")
            para_n_estimators=utils.get_param(params,"n_estimators")
            #パラメタ反映
            model = xgb.XGBClassifier(gamma=float(para_gamma),
                        max_depth=int(para_max_depth),
			min_child_weight=int(para_min_child_weight),
			max_delta_step=int(para_max_delta_step),
			subsample=float(para_subsample),
			#objective=para_objective,
			base_score=float(para_base_score),
			n_estimators=int(para_n_estimators))

    
    #----------読み込み&データ準備---------
    #学習データ
    print " start reading "+TMP_Path_data
    Mat_train = pd.read_csv(TMP_Path_data, encoding ='utf-8',dtype={Sca_flagkey:  object})
    print " end reading "+TMP_Path_data
    
    #----------モデル投入のためのデータ整理----------
    TMP_Vec = np.setdiff1d(Mat_train.columns.values,np.asarray(Vec_exceptcol))
    Vec_label_train = Mat_train[Sca_flagcol]
    Mat_train_key = Mat_train[Sca_flagkey]
    Mat_train = Mat_train[TMP_Vec]
    Mat_train_columns = Mat_train.columns

    #----------モデリング実行----------
    #正規化する場合、コメントイン ↓から
    std_scale = preprocessing.StandardScaler().fit(Mat_train)
    Mat_train = std_scale.transform(Mat_train)
    # ↑まで

    model.fit(Mat_train, Vec_label_train)
    #モデリング結果保存
    with open(TMP_Path_save + TMP_train_filename_head +"_"+learn_m.upper()+'_model.dump','w') as f:
        pickle.dump(model, f)
    #変数名リスト保存
    with open(TMP_Path_save + TMP_train_filename_head +"_"+learn_m.upper()+'_cols.dump','w') as f:
        pickle.dump(Mat_train_columns, f)
    
