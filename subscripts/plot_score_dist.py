# -*- coding: utf-8 -*-
#
# フラグ毎のスコア分布をプロットする
#
# argvs[1]:evaluate.pyによって出力されるprob.csvファイル
# argvs[2]:出力先ディレクトリ
# argvs[3]:評価方法(cv or shft)を指定


#----------基本モジュール----------
import pandas as pd
import numpy as np
import re #ワイルドカード等の正規表現で使用
import sys
#import matplotlib.pyplot as P

import matplotlib
matplotlib.use('Agg')
import pylab

import matplotlib.font_manager#日本語フォントの表示で利用

argvs = sys.argv
in_fn = str(argvs[1])
out_path = str(argvs[2])
eva_method = str(argvs[3])

in_fn_split = re.split("\/", in_fn)
in_fn_base = in_fn_split[len(in_fn_split)-1]
if eva_method == "cv":
	fn = re.split("prob_mean\.csv", in_fn_base)[0]
else:
	fn = re.split("prob\.csv", in_fn_base)[0]

input_data=pd.read_csv(in_fn,encoding="cp932")

data0=input_data[input_data[input_data.columns[1]]==0]
data1=input_data[input_data[input_data.columns[1]]==1]
data0.index=range(0,len(data0))
data1.index=range(0,len(data1))


#fig = plt.figure(figsize=(10,7))
fp = matplotlib.font_manager.FontProperties(fname='./ipaexg.ttf')
vec_bins=range(0,21)*np.repeat(5,21)/np.repeat(100.0,21)

if eva_method == "cv":
	pylab.hist((data0['prob_mean'], data1['prob_mean']), histtype='barstacked', label=('flag 0','flag 1'), bins=vec_bins,color=('b','r'),rwidth=1)
else:
	pylab.hist((data0['prob_1'], data1['prob_1']), histtype='barstacked', label=('flag 0','flag 1'), bins=vec_bins,color=('b','r'),rwidth=1)

pylab.xlim(0,1)
#pylab.ylim(0,500)
pylab.xlabel("score",fontsize=15)
pylab.ylabel("Frequency",fontsize=15)
pylab.title(u"フラグ毎のスコア分布",fontsize=15,fontproperties = fp)
pylab.legend(loc="best")

pylab.savefig(out_path+"/"+fn+"_score_dist.png")

