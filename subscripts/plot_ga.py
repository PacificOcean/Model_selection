# -*- coding: utf-8 -*-
#
# GAの経過をプロットする。1点は1populationを表す
#
# argvs[1]:run_ga.pyで出力されるログファイル
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

in_fn_split = re.split("\/", in_fn)
in_fn_base = in_fn_split[len(in_fn_split)-1]
fn = re.split("\.log", in_fn_base)[0]

ld = open(in_fn)
lines = ld.readlines()
ld.close()

score_list=[]
for line in lines:
    if line.find("self.fit : ") >= 0:
        score_list.append(re.split("self\.fit : ", line[:-1])[1])
        
scores=pd.DataFrame(score_list).astype(float)

fp = matplotlib.font_manager.FontProperties(fname='./ipaexg.ttf')
pylab.plot(scores,'.')
pylab.title(u"精度の変化プロット",fontsize=15,fontproperties = fp)
#plt.xlim([850,1900])
#pylab.ylim([0,1])
pylab.xlabel(u"試行回数",fontproperties = fp)
pylab.ylabel(u"score")


pylab.savefig(out_path+"/"+fn+"_gaplot.png")
