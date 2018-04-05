# -*- coding: utf-8 -*-

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import math
import re

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

#----------精度指標整理関数----------
def Func_predclass_summary(Vec_label_test, output, Mat_predprob = np.array([]), index="None"):
    ##########
    #Vec_label_test:真のラベル、output:予測結果（ラベル）
    #★ラベルはint / floatで 0, 1, 2, ....の値をとることを前提とする。
    #Mat_predprob：所属確率マトリクス（sklearnの.predict_proba）、index:出力レコードの識別キー
    ##########

    ###ラベルがとりうる値の確認（簡易チェック）
    TMP_Sca_min = min(Vec_label_test.min(), output.min())
    TMP_Sca_max = max(Vec_label_test.max(), output.max())
    if TMP_Sca_min != int(TMP_Sca_min): return ValueError #最小値がint/x.000...か確認
    if TMP_Sca_max != int(TMP_Sca_max): return ValueError #最大値がint/x.000...か確認

    TMP_Vec = np.arange(TMP_Sca_min, TMP_Sca_max+1) #ラベルがとりうる値の集合
    if np.asarray(Vec_label_test).shape[0] != np.in1d(np.asarray(Vec_label_test), TMP_Vec).sum(): return ValueError #取りうる値以外が入っていないか確認
    if np.asarray(output).shape[0] != np.in1d(np.asarray(output), TMP_Vec).sum(): return ValueError #取りうる値以外が入っていないか確認


    Mat_ret = DataFrame();  Mat_ret["index"] = Series(index)
    Mat_ret["Accuracy"] = accuracy_score(Vec_label_test, output)

    ###精度指標（平均）算出
    TMP_Mat = precision_recall_fscore_support(Vec_label_test, output, average="binary")
    Mat_ret = pd.concat([Mat_ret, DataFrame([TMP_Mat[:3]], columns=[x + "_ave_binary" for x in ["Precision", "Recall", "F1measure"]])], axis=1)

    ###AUC算出
    if Mat_predprob.shape[0] != 0:
        for i in range(0, Mat_predprob.shape[1]):
            TMP_precision, TMP_recall, TMP_threshold = precision_recall_curve(Vec_label_test, Mat_predprob[:, i], pos_label=i)
            Mat_ret["AUC_PR_"+str(i)] = auc(TMP_recall, TMP_precision)
            TMP_fp, TMP_tp, TMP_threshold = roc_curve(Vec_label_test, Mat_predprob[:, i], pos_label=i)
            Mat_ret["AUC_ROC_"+str(i)] = auc(TMP_fp, TMP_tp)

    ###精度指標（クラス毎）算出
    TMP_Mat = precision_recall_fscore_support(Vec_label_test, output)
    for i, TMP_index in enumerate(["Precision", "Recall", "F1measure", "Support"]):
        Mat_ret = pd.concat([Mat_ret, DataFrame(TMP_Mat[i], index=[TMP_index+"_"+ str(x) for x in list(np.arange(TMP_Mat[i].shape[0]))]).T], axis=1)

    ###confusion_matrix算出
    TMP_Mat = confusion_matrix(Vec_label_test, output)
    TMP_Vec = confusion_matrix(Vec_label_test, output).flatten()
    TMP_Vec_2 = np.asarray([["CM_true"+str(r)+"_pred"+str(c) for c in np.arange(0, TMP_Mat.shape[1])] for r in  np.arange(0, TMP_Mat.shape[0])]).flatten()
    Mat_ret = pd.concat([Mat_ret, DataFrame(TMP_Vec, index=[TMP_Vec_2]).T], axis=1)

    return Mat_ret

#----------非復元抽出サンプリング関数----------
def Func_sampling_nodup(a, n):
    ##########
    #a:入力ベクトル、n:サンプリング数
    ##########
    for i in range(0, n):
        index = random.randint(i,len(a)-1)#インデックス生成
        a[i], a[index] = a[index], a[i]   #インデックスとサンプリング済みの要素をスワップ
    return a[:n]

#----------学習器パラメタ取得関数----------
def get_param(para_list,key):
    ret_val=""
    for tmp_para in para_list:
        tmp_str=re.split("\=", tmp_para)
        if tmp_str[0]==key:
            ret_val=tmp_str[1]

    #チェック
    if ret_val=="":
        print "WARNGING: parameter does not match.: "+key
    return ret_val

#---------文字列検索関数----------------
def grep(files,word):
    tgt = []
    for file in files:
        if file.find(word) >= 0:
            tgt.append(file)
    return tgt

def grep_startswith(lst,word):
    tgt = []
    for string in lst:
        if string.startswith(word) == True:
            tgt.append(string)
    return tgt

def grep_endswith(lst,word):
    tgt = []
    for string in lst:
        if string.endswith(word) == True:
            tgt.append(string)
    return tgt
