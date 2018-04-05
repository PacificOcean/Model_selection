# -*- coding: utf-8 -*-
#予測の0／1を決めるための閾値を0.1～0.9まで動かした時の、精度指標とconfusion_matrixの要素を出力
#閾値0.5の場合も、スコアの平均値をもとに精度指標を再計算しているため、精度指標の平均値が出力されているevamean_summary.csvと値が異なる
#複数の結果をマージする版　別々のモデルで出力されることを想定
#
# argvs[1]: evaluate.pyで出力したprob_mean.csvファイルのパス カンマで区切りで複数指定可能
# argvs[2]: 目的変数のカラム名
# argvs[3]: 出力先ディレクトリ
# argvs[4]: 評価方法(cv or shft)を指定

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import re
import sys
from numpy import nan as NA

argvs = sys.argv


in_fn = str(argvs[1])
flag_col = str(argvs[2])
output_path = str(argvs[3])
eva_method = str(argvs[4])

flag_col = flag_col.decode('utf-8')

Vec_in_fn=re.split(",", in_fn)

input_data=pd.DataFrame()
fname_head=""
for in_fn in Vec_in_fn:
   tmp_data=pd.read_csv(in_fn,encoding="cp932")#,dtype={'ID': object})
   input_data=pd.concat([input_data,tmp_data])
   fname_split1 = re.split("\/", in_fn)
   if eva_method=="cv":
      fname_split2 = re.split("prob_mean\.csv", fname_split1[len(fname_split1)-1])
   else:
      fname_split2 = re.split("prob\.csv", fname_split1[len(fname_split1)-1])
   tmp_fname_head=fname_split2[0]
   if len(fname_head)==0:
      fname_head=tmp_fname_head
   else:
      fname_head=fname_head+"_"+tmp_fname_head

out_df=pd.DataFrame()
for threshold in np.arange(0.05,1,0.05):
	if eva_method=="cv":
		pred1_data=input_data[input_data["prob_mean"]>=threshold]
		pred0_data=input_data[input_data["prob_mean"]<threshold]
	else:
		pred1_data=input_data[input_data["prob_1"]>=threshold]
		pred0_data=input_data[input_data["prob_1"]<threshold]
	try: a=pred1_data[flag_col].value_counts().ix[1,1]
	except: a=0
	try: b=pred0_data[flag_col].value_counts().ix[1,1]
	except: b=0
	try: c=pred1_data[flag_col].value_counts().ix[0,1]
	except: c=0
	try: d=pred0_data[flag_col].value_counts().ix[0,1]
	except: d=0

	if a+c==0: precision=NA
	else: precision=float(a)/(a+c)
	if a+b==0: recall=NA
	else: recall=float(a)/(a+b)
	if (recall+precision==0): recall=NA
	else: F_m=float(2*recall*precision)/(recall+precision)
	Acu=float(a+d)/(a+b+c+d)
        if c+d==0: specificity=NA
        else: specificity=float(d)/(c+d)

	tmp_df=pd.DataFrame([fname_head,threshold,Acu,precision,recall,specificity,F_m,a,b,c,d]).T
	out_df=pd.concat([out_df,tmp_df])

out_df.columns=["file_name","threshold","Accuracy","precision","recall","specificity","F-measure","CM_true1_pred1","CM_true1_pred0","CM_true0_pred1","CM_true0_pred0"]

out_df.to_csv(output_path+"/"+fname_head+"confusion_matrix.csv",index=False, encoding="cp932")

