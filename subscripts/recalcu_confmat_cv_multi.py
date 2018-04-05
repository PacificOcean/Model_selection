# -*- coding: utf-8 -*-
#CV式で求めた多値分類の予測ラベル（平均値）から、予測値を決定する。
#予測値を決定した後の精度指標とconfusion_matrixの要素を出力する。
#複数の結果をマージする版　別々のモデルで出力されることを想定
#
# argvs[1]: evaluate_by_cv.pyで出力したprob_mean.csvファイルのパス カンマで区切りで複数指定可能
# argvs[2]: 目的変数のカラム名
# argvs[3]: 出力先ディレクトリ

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import re
import sys
from numpy import nan as NA

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import utils

argvs = sys.argv

in_fn = str(argvs[1])
flag_col = str(argvs[2])
output_path = str(argvs[3])

flag_col = flag_col.decode('utf-8')

Vec_in_fn=re.split(",", in_fn)


#---------文字列検索関数----------------
def extract_flag(string):
    return re.split("_mean",re.split("prob_",string)[1])[0]


input_data=pd.DataFrame()
fname_head=""
for in_fn in Vec_in_fn:
   tmp_data=pd.read_csv(in_fn,encoding="cp932")#,dtype={'ID': object})
   input_data=pd.concat([input_data,tmp_data])
   fname_split1 = re.split("\/", in_fn)
   fname_split2 = re.split("prob_mean\.csv|prob\.csv", fname_split1[len(fname_split1)-1])
   tmp_fname_head=fname_split2[0]
   if len(fname_head)==0:
      fname_head=tmp_fname_head
   else:
      fname_head=fname_head+"_"+tmp_fname_head


test_flag=input_data[flag_col].astype(int)

pred_columns=utils.grep_endswith(utils.grep_startswith(input_data.columns, "prob_"),"_mean")
pred_data=input_data[pred_columns]
pred_flag_str=pred_data.T.idxmax()
pred_flag=pred_flag_str.apply(extract_flag).astype(int)

Mat_ret = pd.DataFrame()
Mat_ret["file_name"] = Series(fname_head)
Mat_ret["Accuracy"] = accuracy_score(test_flag, pred_flag)
TMP_Mat = precision_recall_fscore_support(test_flag, pred_flag, average="binary")
Mat_ret = pd.concat([Mat_ret, DataFrame([TMP_Mat[:3]], columns=[x + "_ave_binary" for x in ["Precision", "Recall", "F1measure"]])], axis=1)


TMP_Mat = precision_recall_fscore_support(test_flag, pred_flag)
for i, TMP_index in enumerate(["Precision", "Recall", "F1measure", "Support"]):
        Mat_ret = pd.concat([Mat_ret, DataFrame(TMP_Mat[i], index=[TMP_index+"_"+ str(x) for x in list(np.arange(TMP_Mat[i].shape[0]))]).T], axis=1)

TMP_Mat = confusion_matrix(test_flag, pred_flag)
TMP_Vec = confusion_matrix(test_flag, pred_flag).flatten()
TMP_Vec_2 = np.asarray([["CM_true"+str(r)+"_pred"+str(c) for c in np.arange(0, TMP_Mat.shape[1])] for r in  np.arange(0, TMP_Mat.shape[0])]).flatten()
Mat_ret = pd.concat([Mat_ret, DataFrame(TMP_Vec, index=[TMP_Vec_2]).T], axis=1)

Mat_ret.to_csv(output_path+"/"+fname_head+"confusion_matrix.csv",index=False, encoding="cp932")

