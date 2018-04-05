#-*- coding:utf-8 -*-

###################################################
# GAによるパラメタ探索
###################################################
# sample.rf.pyを改造
# 
# データ：
#  カラム名、ID識別キー、説明変数を含んだままのデータ(utf-8指定)をそのまま使う
# 学習器：
#  rf,svm,logisticに対応
#
# 引数：
#  argvs[1]:データファイルを指定。
#  argvs[2]:ID識別キーのカラム名を指定。
#  argvs[3]:目的変数のカラム名を指定
#  argvs[4]:GAの探索で使用するpopulationを指定。
#  argvs[5]:GAの探索で使用するgenerationを指定。
#  argvs[6]:GAの探索で使用するseedを指定。
#  argvs[7]:学習器を指定。 rf,svm,lnsvm,logistic,xgb

import csv
import numpy as np
import pandas as pd
import random
from collections import OrderedDict
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from sklearn import cross_validation
from sklearn import preprocessing

import os
os.environ['JOBLIB_TEMP_FOLDER'] = "/var/"

import ga

import sys
argvs = sys.argv

#---------------------------------------------
#データ読み込み部分
#X:説明変数
#y:目的変数
#---------------------------------------------

fn = str(argvs[1])
index_col = str(argvs[2])
flag_col = str(argvs[3])
ga_population = int(argvs[4])
ga_generation = int(argvs[5])
ga_seed = int(argvs[6])
learn_m = str(argvs[7])

train = pd.read_csv(fn,encoding="utf-8")
flag = train[flag_col]

#flag列を除く
del train[flag_col]

#indexを除く
if index_col=="1":
	index = train.ix[:,0]
	train = train.ix[:,1:]
elif index_col=="0":
	dummy=1
else:
	del train[index_col]

#正規化
std_scale = preprocessing.StandardScaler().fit(train)
train = std_scale.transform(train)

#np.arrayに変換
X = np.array(train)
y = np.array(flag)

#flagを0/1に変換
#y[y=='spam'] = '1'
#y[y=='nonspam'] = '0'
#y[y=='0.0'] = '1'
#y[y=='1.0'] = '0'
y = y.astype(int)


#---------------------------------------------
#model_obj : 探索したいモデルのオブジェクト
#score_func : 最適化する目的関数の設定を行う(下記はcross_validationの使用例)
#
# スコア関数は説明変数(X),目的変数(y),
# モデル(model)の引数位置を指定するために必要
#---------------------------------------------

if learn_m=='rf':
	model_obj = RandomForestClassifier()# RandomForest以外も使用可能
elif learn_m=='lnsvm':
	model_obj = LinearSVC()
elif learn_m=='svm':
	model_obj = SVC()
elif learn_m=='logistic':
        model_obj = LogisticRegression()
elif learn_m=='xgb':
        model_obj = xgb.XGBClassifier()
else:
	exit(1)

# 通常の使用ではscoreing,n_jobsを変更するだけでOK
if learn_m=='rf': #RFの場合は1にして、RFのn_jobsで並列数を決める
        cv_n_jobs=1
else:
        cv_n_jobs=-1
score_func = lambda model,X,y: np.mean(cross_validation.cross_val_score(
                    estimator = model,
                    X=X,
                    y=y,
                    cv=5,
                    scoring='f1', # 使いたいスコアリング関数(sklearn参照)
                    #'roc_auc','accuracy','precision','average_precision','recall','f1'
                    n_jobs=cv_n_jobs # cross_validationの並列数
                    #n_jobs=1 # cross_validationの並列数
                )
            )


#---------------------------------------------
#RandomForestで探索するパラメータの範囲を指定する
#dictでも渡せるが、遺伝子内の変数とparamの設定順序が見かけ上対応しなくなるので、
# OrderedDictの使用を推奨
#離散パラメータ:intのtuple
#連続パラメータ:floatのtuple
#固定したいパラメータ:int or float or str
#設定していないパラメータ:デフォルト引数が使用される
#---------------------------------------------
model_params = OrderedDict()
if learn_m=='rf':
	model_params['n_estimators'] = (10,300) # 1以上400以下 (int)
	model_params['max_features'] = (0.0,1.0) # 0.0以上1.0未満 (float)
	model_params['max_depth'] = (10,80) #(1,100)
	model_params['min_samples_split'] = (1,100) #(1,100)
	model_params['n_jobs'] = -1 #2 #int の 2 に固定 (RandomForestの並列数)
	model_params['min_samples_leaf'] = (1,10)
	model_params['random_state'] = 1

elif learn_m=='lnsvm':
	model_params['C'] = (0.0,1.0)
	model_params['tol'] = (0.0,1.0)
	#model_params['loss'] = 'hinge'
	model_params['random_state'] = 1

elif learn_m=='svm':
	model_params['C'] = 100000
	#model_params['degree'] = (1,10)
	#model_params['gamma'] = (0.0,1.0)
	#model_params['coef0'] = (0.0,1.0)
	#model_params['tol'] = (0.0,1.0)
	model_params['random_state'] = 1
        #model_params['probability'] = True

elif learn_m=='logistic':
        model_params['C'] = (0.1,10.0) #1.0
	model_params['random_state'] = 1
elif learn_m=='xgb':
	#model_params['eta'] = (0.0,1.0)
	model_params['gamma'] = (0.0,5.0)
	model_params['max_depth'] = (1,20)
	model_params['min_child_weight'] = (0,4)
	model_params['max_delta_step'] = (0,9)
	model_params['subsample'] = (0.0,1.0)
	#model_params['num_round'] = (1,20)
	#model_params['silent'] = 1
	#model_params['objective'] = 'binary:logistic'
	model_params['base_score'] = (0.0,1.0)
	model_params['n_estimators'] = (10,300) # 1以上400以下 (int)
else:
	exit(1)

# ※最終的な並列数は(cross_validationの並列数)×(モデルの並列数)になるので注意

#---------------------------------------------
#GAで使用するパラメータ
#population:1世代の個体数
#generation:世代数
#mutation_g:遺伝子の突然変異確率
#mutation_p:突然変異する遺伝子が変更をこうむる割合
#seed:GAの探索で使用するシード
#
# ※(計算時間) = (cross_validation　1回でかかる時間) × population × generation
# population と generation は同じくらいの値に設定するとよい
# population と generation を大きくするほどよい探索結果になる
#---------------------------------------------
population = ga_population
generation = ga_generation
mutation_g = 0.2 # 基本的にこの値でままで良い
mutation_p = 0.05 # 基本的にこの値のままで良い
seed = ga_seed
#seed = 1 #2

#---------------------------------------------
# GAによる変数＆パラメータ探索の実行
#
#(1)GAオブジェクト作成時に、パラメータ群、X(説明変数),y(目的変数)を引数にする
#(2)main()メソッドで探索実行 (計算の過程は標準出力に出る)
#(3)(2)のreturnは[スコア,[遺伝子],[選択されたカラム番号],{選択されたパラメータの値}]
#  - 例 : [0.8,[0,1,0,1,0.5],[1,3],{'max_features':0.5}]
#---------------------------------------------
ga_list = [] # 結果格納用
for s in range(seed):# seed 0,1 で実行
    print '=== ga seed : {} ==='.format(str(s))
    # GAオブジェクトの初期設定
    optga = ga.ga(model_obj,score_func,model_params,population,generation,mutation_g,mutation_p,s,X,y)
    # 探索実行し、結果をリストに格納
    ga_list += [optga.main()]

# 一応結果のリストを保存しておく
with open('./rf_ga_list.dump','w') as f:
    pickle.dump(ga_list, f)


