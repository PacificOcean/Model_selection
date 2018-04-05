# -*- coding: utf-8 -*-
#
# 入力データの中で、学習データ、テストデータの組み合わせを変えて、評価をn回実施した平均値を出力する。
# また、重要度の低い変数を取り除いたデータを作成する（オプション）
# データが大きい場合に、GAを実行する前に、これである程度変数を絞る。
#
# インプット：CSV形式で、カラム名あり。識別キー、目的変数、説明変数を含む。
# アウトプット：予測スコア、重要度、評価指標
#               重要度が0でないカラムのみの学習データ、テストデータ、カラムリスト
#
# argvs[1]:入力データ (カンマ区切りでリストも可)
# argvs[2]:GAの結果のパラメータデータ (カンマ区切りでリストも可) #GA無しの場合は"0"とする。
# argvs[3]:ID識別キーのカラム名を指定
# argvs[4]:目的変数のカラム名を指定
# argvs[5]:評価出力先ディレクトリ
# argvs[6]:学習・テストの繰り返し回数n
# argvs[7]:学習・テストの分割数。(m-1):1となる。
# argvs[8]:学習器 rf,svm,logistic,xgb
# argvs[9]:重要度の低いカラムを除いたデータを作成するかどうか。0:しない、1:する
# argvs[10]:テストデータ (カンマ区切りでリストも可)を指定。存在しない場合は0。
#           重要度の低いカラムをテストデータからも除くために指定する。

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
from sklearn.cross_validation import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
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
seed_num = int(argvs[6])
foldnum = int(argvs[7])
learn_m = str(argvs[8])
extract_impcol = str(argvs[9])
test_fn = str(argvs[10])

index_col = index_col.decode('utf-8')
flag_col = flag_col.decode('utf-8')

#----------パス関連---------
Vec_Path_data=re.split(",", in_fn)
#Vec_Path_data = [in_fn]#入力ファイルリスト

Vec_Path_test_data=re.split(",", test_fn)

#GAで選択されたパラメタのデータ
if param_fn != "0":
    Vec_Path_paramdata = re.split(",", param_fn)
    #Vec_Path_paramdata = [param_fn]

#出力先ディレクトリ
Path_rootdir_out = out_path +"/"

#----------データ関連---------
Sca_flagcol = flag_col #正解フラグの入っている変数名
Sca_flagkey = index_col #サンプルの識別キーが入っている変数名
Vec_exceptcol = [Sca_flagcol, Sca_flagkey] #推定に使わない変数名

Sca_2div_ration = 0.5 #学習で使用するサンプルの割合 #2分割の時にのみ有効。固定。
Vec_seed = np.arange(0, seed_num) #サンプル抽出時のseed
Sca_ccode = "cp932" #出力の文字コード

#----------出力関連---------
Flag_out_perseed = True #False #seed毎の結果を算出する
Flag_out_perseed_evasum = True #False #Path_rootdir_out直下にモデルの精度を1つにまとめたファイル(これはcp932固定)を作成
#ファイルが既に存在する場合は、そのファイルを消さず追記していく（変数名は最初のレコードに基づく）。

Flag_out_evamean = True #ファイル毎に各seed設定の精度評価値を平均したものを出力する
Flag_out_evamean_evasum = True


#####予測実行##### 　各指標はCV毎の平均値とする 一時ディレクトリ使用しない版 パラレルなし版
loop_cnt=-1
for TMP_Path_data in Vec_Path_data:
    #print TMP_Path_data
    loop_cnt=loop_cnt+1
    #----------パス整理---------
    TMP_Path_data_split = re.split("\/", TMP_Path_data)
    TMP_filename_head = re.split("\.csv", TMP_Path_data_split[len(TMP_Path_data_split)-1])[0]
    #TMP_filename_head = unicode(TMP_filename_head, "utf-8")
    TMP_Path_save = Path_rootdir_out

    #----------学習器パラメタ設定---------
    if learn_m=="rf":
        #GAなし
        if param_fn == "0":
            #model = RandomForestClassifier(n_jobs=-1, random_state=1)#, class_weight={0:1, 1:40})
            #model = RandomForestClassifier(n_jobs=-1,n_estimators=100,random_state=1,max_features=0.3)
            model = RandomForestClassifier(n_jobs=-1,
                                           n_estimators=100,
                                           max_depth=50,
                                           min_samples_split=50,
                                           min_samples_leaf=10,
                                           random_state=1)
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
    
    #----------読み込み&データ準備 分割---------
    print " start reading "+TMP_Path_data
    #Mat_input = pd.read_csv(TMP_Path_data, encoding ='utf-8',dtype={Sca_flagkey: int})
    Mat_input = pd.read_csv(TMP_Path_data, encoding ='utf-8',dtype={Sca_flagkey: object})
    print " end reading "+TMP_Path_data
    
    all_index = Mat_input.index
      
    Mat_RFeva_mean = DataFrame()
    Mat_RFimp_mean = DataFrame()
    Mat_RFprob_mean = DataFrame()    
    
    #seedで分割を変更してループ
    for TMP_seed in Vec_seed:
        print " "+str(TMP_seed+1)+"/"+str(seed_num)
        list_index = StratifiedKFold(Mat_input[Sca_flagcol], n_folds=foldnum, shuffle=True)#, random_state=1)
        cv_num=len(list_index)
        #テストデータを変えてループ(CV)
        cv_cnt=-1
        for tmp_lst_ix in list_index:
            cv_cnt=cv_cnt+1
            #tmp_lst_ixをテストで使用して、tmp_lst_ix以外を学習で使用する
            Mat_train = Mat_input.ix[tmp_lst_ix[0]]
            Vec_label_train = Mat_train[Sca_flagcol]
            TMP_Vec = np.setdiff1d(Mat_train.columns.values,np.asarray(Vec_exceptcol))
            Mat_train_key = Mat_train[Sca_flagkey]
            Mat_train = Mat_train[TMP_Vec]

            Mat_test = Mat_input.ix[tmp_lst_ix[1]]
            Vec_label_test = Mat_test[Sca_flagcol]
            Mat_test_key = Mat_test[Sca_flagkey]
            Mat_test = Mat_test[TMP_Vec]

            #----------モデリング実行----------
            #正規化
            Mat_train_columns=Mat_train.columns
            std_scale = preprocessing.StandardScaler().fit(Mat_train)
            Mat_train = std_scale.transform(Mat_train)
            Mat_train = pd.DataFrame(Mat_train)
            Mat_train.columns=Mat_train_columns
            Mat_test_columns=Mat_test.columns
            Mat_test = std_scale.transform(Mat_test)
            Mat_test = pd.DataFrame(Mat_test)
            Mat_test.columns=Mat_test_columns

            model.fit(Mat_train, Vec_label_train)
            output = model.predict(Mat_test)

            #----------結果整理・保存----------
            ###各種精度
            Mat_RFeva = utils.Func_predclass_summary(Vec_label_test, output, Mat_predprob = model.predict_proba(Mat_test), index=TMP_filename_head+"_cv"+str(cv_cnt)+"_seed"+str(TMP_seed))

            if Flag_out_perseed == True:
                ###重要度
                if learn_m=="rf":
                    Mat_RFimp = DataFrame(np.c_[Mat_test.columns.values, model.feature_importances_], columns=["variable", "importance"+"_cv"+str(cv_cnt)+"_seed"+str(TMP_seed)])
                if learn_m=="logistic": #係数
                    Mat_RFimp = DataFrame(np.c_[np.r_[Mat_test.columns.values, np.asarray(["intercept"], dtype="object")], np.c_[model.coef_, model.intercept_].transpose(1,0)], columns=["variable", "importance"+"_cv"+str(cv_cnt)+"_seed"+str(TMP_seed)])

                ###ラベル確率
                TMP_Mat = np.c_[Mat_test_key.values, Vec_label_test.values]
                TMP_Mat_2 = model.predict_proba(Mat_test)
                Mat_RFprob = DataFrame(np.c_[TMP_Mat, TMP_Mat_2], columns=[Sca_flagkey, Sca_flagcol]+["prob_" + str(i) for i in list(np.arange(TMP_Mat_2.shape[1]))])
                if len(np.arange(TMP_Mat_2.shape[1]))==2: #2値の場合は、1となる確率のみを出力
                    Mat_RFprob = DataFrame(np.c_[TMP_Mat, TMP_Mat_2[:,1]], columns=[Sca_flagkey, Sca_flagcol,"prob_1"+"_cv"+str(cv_cnt)+"_seed"+str(TMP_seed)])
                    #Mat_RFprob[Sca_flagkey]=Mat_RFprob[Sca_flagkey].astype(int)
                else:
                    #Mat_RFprob = DataFrame(np.c_[TMP_Mat, TMP_Mat_2], columns=[Sca_flagkey, Sca_flagcol]+["prob_" + str(i) for i in list(np.arange(TMP_Mat_2.shape[1]))])
                    Mat_RFprob = DataFrame(np.c_[TMP_Mat, TMP_Mat_2], columns=[Sca_flagkey, Sca_flagcol]+["prob_" + str(i) +"_cv"+str(cv_cnt)+"_seed"+str(TMP_seed) for i in list(np.arange(TMP_Mat_2.shape[1]))])
                    #Mat_RFprob[Sca_flagkey]=Mat_RFprob[Sca_flagkey].astype(int)

            if Flag_out_evamean == True:
                Mat_RFeva_mean = pd.concat([Mat_RFeva_mean, Mat_RFeva], axis=0)

                if (learn_m=="rf")|(learn_m=="logistic"):
                    if len(Mat_RFimp_mean)==0:
                        Mat_RFimp_mean=Mat_RFimp
                    else:
                        Mat_RFimp_mean = pd.merge(Mat_RFimp_mean,Mat_RFimp,on='variable')
                if len(np.arange(TMP_Mat_2.shape[1]))==2: #2値の場合は、1となる確率のみを出力
                    if len(Mat_RFprob_mean)==0:
                        Mat_RFprob_mean=Mat_RFprob[[Sca_flagkey,Sca_flagcol,'prob_1'+"_cv"+str(cv_cnt)+"_seed"+str(TMP_seed)]]
                    else:
                        Mat_RFprob_mean = pd.merge(Mat_RFprob_mean,Mat_RFprob[[Sca_flagkey,Sca_flagcol,'prob_1'+"_cv"+str(cv_cnt)+"_seed"+str(TMP_seed)]],on=[Sca_flagkey,Sca_flagcol],how='outer')
                else: #多値の場合
                    for i in list(np.arange(TMP_Mat_2.shape[1])):
                        if len(Mat_RFprob_mean)==0:
                            Mat_RFprob_mean=Mat_RFprob[[Sca_flagkey,Sca_flagcol,'prob_'+ str(i) +"_cv"+str(cv_cnt)+"_seed"+str(TMP_seed)]]
                        else:
                            Mat_RFprob_mean = pd.merge(Mat_RFprob_mean,Mat_RFprob[[Sca_flagkey,Sca_flagcol,'prob_'+ str(i) +"_cv"+str(cv_cnt)+"_seed"+str(TMP_seed)]],on=[Sca_flagkey,Sca_flagcol],how='outer')
                    
    if Flag_out_evamean == True:
        Mat_RFeva_mean = DataFrame(Mat_RFeva_mean[Mat_RFeva_mean.columns[Mat_RFeva_mean.columns != "index"]].mean(0)).T
        Mat_RFeva_mean_2 = DataFrame();  Mat_RFeva_mean_2["index"] = Series(TMP_filename_head+"_mean_cv"+str(cv_num))
        Mat_RFeva_mean_2 = pd.concat([Mat_RFeva_mean_2, Mat_RFeva_mean], axis=1)
        Mat_RFeva_mean_2.to_csv(TMP_Path_save + TMP_filename_head+"_"+learn_m.upper()+"eva_mean.csv", index=False, encoding=Sca_ccode)

        if (learn_m=="rf")|(learn_m=="logistic"):
            Mat_RFimp_mean['importance_mean']=Mat_RFimp_mean.ix[:,1:].T.mean()
            Mat_RFimp_mean.to_csv(TMP_Path_save + TMP_filename_head+"_"+learn_m.upper()+"imp_mean.csv", index=False, encoding=Sca_ccode)
        
        if len(np.arange(TMP_Mat_2.shape[1]))==2: #2値の場合は、1となる確率のみを出力
            Mat_RFprob_mean['prob_mean']=Mat_RFprob_mean.ix[:,2:].T.mean()
        else:  #多値の場合
            for i in list(np.arange(TMP_Mat_2.shape[1])):
                tmp_columns=utils.grep(Mat_RFprob_mean.columns, "prob_"+str(i)+"_cv")
                Mat_RFprob_mean['prob_'+str(i)+'_mean']=Mat_RFprob_mean[tmp_columns].T.mean()

        Mat_RFprob_mean=Mat_RFprob_mean.sort_values(by=Sca_flagkey)
        #Mat_RFprob_mean[Sca_flagkey]=Mat_RFprob_mean[Sca_flagkey].astype(int)
        Mat_RFprob_mean.to_csv(TMP_Path_save + TMP_filename_head+"_"+learn_m.upper()+"prob_mean.csv", index=False, encoding='utf-8')
                
        if Flag_out_evamean_evasum == True:
            TMP_Path = Path_rootdir_out + "/evamean_summary.csv"
            if not(os.path.exists(TMP_Path)): Mat_RFeva_mean_2.to_csv(TMP_Path, index=False, encoding=Sca_ccode, mode='a')
            else: Mat_RFeva_mean_2.to_csv(TMP_Path, index=False, encoding=Sca_ccode, mode='a', header=None)

    if extract_impcol == "1":
        #n回の評価で、重要度が全て0の変数を元のデータから除く
        after_imp="_imp"
        imp_data=Mat_RFimp_mean.copy()
        imp_data2=imp_data[[u'variable',u'importance_mean']]
        #imp_data3=imp_data2.sort('importance_mean', ascending=False)
        #imp_data3.index=range(0,len(imp_data3))
        imp_data3=imp_data2[imp_data2['importance_mean']>0]
        impcollist=[Sca_flagkey]
        impcollist.extend(list(imp_data3['variable']))
        impcollist.append(Sca_flagcol)
        if "intercept" in impcollist:
            impcollist.remove("intercept")

        TMP_Path_data_rmcsv = re.split("\.csv", TMP_Path_data)[0]
        print " start creating "+TMP_Path_data_rmcsv+after_imp+".csv"
        Mat_input[impcollist].to_csv(TMP_Path_data_rmcsv+after_imp+".csv",encoding='utf-8',index=False)
        print " end creating "+TMP_Path_data_rmcsv+after_imp+".csv"

        if (Vec_Path_test_data[loop_cnt] == Vec_Path_test_data[loop_cnt])&(Vec_Path_test_data[loop_cnt] != "0"):
            print " start reading "+Vec_Path_test_data[loop_cnt]
            #Mat_test = pd.read_csv(Vec_Path_test_data[loop_cnt], encoding ='utf-8',dtype={Sca_flagkey: int})
            Mat_test = pd.read_csv(Vec_Path_test_data[loop_cnt], encoding ='utf-8',dtype={Sca_flagkey: object})
            print " end reading "+Vec_Path_test_data[loop_cnt]

            TMP_Path_testdata_rmcsv = re.split("\.csv", Vec_Path_test_data[loop_cnt])[0]
            print " start creating "+TMP_Path_testdata_rmcsv+after_imp+".csv"
            Mat_test[impcollist].to_csv(TMP_Path_testdata_rmcsv+after_imp+".csv",encoding='utf-8',index=False)
            print " end creating "+TMP_Path_testdata_rmcsv+after_imp+".csv"


