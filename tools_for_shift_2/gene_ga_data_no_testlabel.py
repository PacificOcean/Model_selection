#-*- coding:utf-8 -*-
#
# GAの結果から、最良の変数を抽出したデータを出力する。
#
# argvs[1]: 学習データ。GAのインプットにしたファイル
# argvs[2]: テストデータ。このデータから最良の変数を抽出する。学習データを指定することも可能。
#           その場合、学習データに対して、変数選択されたデータを出力する。
# argvs[3]: カラム番号のリスト。GAのアウトプット
# argvs[4]: ID識別キー 存在しない場合は0。
# argvs[5]: フラグ名
# argvs[6]: 指定しない。現状は、同じディレクトリに出力する
#           ファイル名は、「元のファイル名+_ga_学習器」

import numpy as np
import pandas as pd
import re
import sys
import csv

argvs = sys.argv

in_fn = str(argvs[1])
test_fn = str(argvs[2])
cols_fn = str(argvs[3])
index_col = str(argvs[4])
flag_col = str(argvs[5])
#out_fn = str(argvs[6])

cols_fn_split = re.split("\/", cols_fn)
cols_fn_base = cols_fn_split[len(cols_fn_split)-1]
method = re.split("\_", cols_fn_base)[2]
out_fn = re.split("\.csv", test_fn)[0]+"_ga_"+method+".csv"
out_colname = re.split("\.csv", test_fn)[0]+"_ga_"+method+"_colnames.csv"

#元ファイル読み込み
train = pd.read_csv(in_fn,encoding="utf-8")
test = pd.read_csv(test_fn,encoding="utf-8")

#trainからflag列を除く
del train[flag_col]

#trainからindexを除く
if index_col != "0":
	del train[index_col]

#最適カラム取得
colnumbers=np.loadtxt(cols_fn, delimiter=',',dtype=int)
colnames=train.columns[colnumbers.tolist()]

#DataFrame再構成
test_col_list=[]
if index_col != "0":
	test_col_list = [index_col]
test_col_list.extend(colnames)
#test_col_list.append(flag_col)

output_test=test[test_col_list]

#出力
#データ
output_test.to_csv(out_fn,index=False,encoding='utf-8')

#カラム名
f = open(out_colname, 'wb')
test_col_list_cp932 = [ v.encode('cp932') for v in test_col_list ]
csvWriter = csv.writer(f)
csvWriter.writerow(test_col_list_cp932)
f.close()

