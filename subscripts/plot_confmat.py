# -*- coding: utf-8 -*-
#
# 閾値を変化させたときの各指標値をプロットする
#
# argvs[1]:output_thresh_confmat.pyで出力される混同行列のcsvファイル
# argvs[2]:出力先ディレクトリ



#----------基本モジュール----------
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import os
from os.path import join, relpath #ファイルのパス名の操作で使用
import sys
import math
import re #ワイルドカード等の正規表現で使用
import time
from numpy import nan as NA
import random
import shutil


import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.font_manager#日本語フォントの表示で利用


argvs = sys.argv
in_fn = str(argvs[1])
out_path = str(argvs[2])

#1ファイルについて、各指標値を1枚にプロットする
cm_data = pd.read_csv(in_fn)

in_fn_split = re.split("\/", in_fn)
in_fn_base = in_fn_split[len(in_fn_split)-1]
fn = re.split("confusion_matrix\.csv", in_fn_base)[0]

x_label=u"閾値"
y_label=u"精度"
#グラフ描写
#fig = plt.figure(figsize=(10,7)) #グラフサイズ指定

columns1='precision'
columns2='recall'
columns3='F-measure'
columns4='Accuracy'
columns5='specificity'

y1 = cm_data[columns1]
y2 = cm_data[columns2]
y3 = cm_data[columns3]
y4 = cm_data[columns4]
y5 = cm_data[columns5]
x = cm_data['threshold']

fp = matplotlib.font_manager.FontProperties(fname='./ipaexg.ttf')

pylab.title(u"閾値と精度の関係",fontsize=15, fontproperties = fp)
pylab.plot(x, y1, label=u'Precision')
pylab.plot(x, y2, label=u'Recall')
pylab.plot(x, y3, label=u'F-measure')
pylab.plot(x, y4, label=u'Accuracy')
#pylab.plot(x, y5, label=u'Specificity')
#pylab.legend(bbox_to_anchor=(1.13, 0.17), prop={'size' : 8}) #凡例位置指定
pylab.legend(loc="best", prop={'size' : 8}) #凡例位置指定
pylab.xlabel(x_label,fontsize=15,fontproperties = fp)
pylab.ylabel(y_label,fontsize=15,fontproperties = fp)
pylab.ylim(0, 1)

pylab.savefig(out_path+"/"+fn+"_threshold_plot.png")


