from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier as vfdt
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier as efdt
from catboost import CatBoostClassifier as cgb
from lightgbm import LGBMClassifier as lgb
import xgboost as xgb

from sklearn.datasets import load_svmlight_file
from pathlib import Path
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

DATA_PATH = Path('D://dataset')

def read_libsvm(dname):
    if (DATA_PATH / f'{dname}.txt').exists():
        xs, ys = load_svmlight_file(str((DATA_PATH / f'{dname}.txt')))
        pt = 0.2 if dname != 'iris' else 0.1
        xs, xs_t, ys, ys_t = train_test_split(xs.toarray(), ys, test_size=0.2)
        return xs, ys, xs_t, ys_t
    else:
        dir = DATA_PATH / dname
        xs, ys = load_svmlight_file(str(dir / f'{dname}.txt'))
        xs_t, ys_t = load_svmlight_file(str(dir / f'{dname}.t'))
        return xs.toarray(), ys, xs_t.toarray(), ys_t
    

class OPE:
    def __init__(self):
        self.a = random.randint(1, 1<<2)
        self.b = random.randint(1, 1<<2)
        self.c = random.randint(1, 1<<2)

    def __call__(self, x):
        res = self.a * (x - self.c) ** 2 + self.b
        if x >= self.c:
            return res
        return 2 * self.b - res
        # return 1.5 * x + 2


class mOPE:
    def __init__(self, arr):
        self.arr = sorted(arr)
        self.dict = {i: j for j, i in enumerate(self.arr)}

    def __call__(self, x):
        return self.dict[x]


class Phaser:
    def __init__(self, arr, p):
        m = p * 0.5
        # m = 0.1
        self.mi = min(arr) - m
        self.ma = max(arr) + m
        self.p = p

    def __call__(self, x):
        x = x + self.p
        if x > self.ma:
            return x + self.mi - self.ma
        elif x < self.mi:
            return x + self.ma - self.mi
        else:
            return x

def trans_phase(ds, i):
    A = [x[i] for x in ds[0]]
    r = random.random() * 0.4 + 0.4
    # r = 0.8
    pa = Phaser(A, (max(A)-min(A)) * r)
    for x in ds[0]:
        x[i] = pa(x[i])
    for x in ds[2]:
        x[i] = pa(x[i])

def trans_ope(ds, mope=False):
    nf = len(ds[0][0])
    for i in range(nf):
        if mope:
            _arr1 = [j[i] for j in ds[0]]
            _arr2 = [j[i] for j in ds[2]]
            ope = mOPE(_arr1+_arr2)
        else:
            ope = OPE()
        for x in ds[0]:
            x[i] = ope(x[i])
        for x in ds[2]:
            x[i] = ope(x[i])

def test_md_ds(md, ds):
    if md == xgb:
        clf = xgb.XGBRegressor(tree_method="gpu_hist")
        clf.fit(ds[0], ds[1])
    elif md == lgb:
        clf = lgb()
        clf.fit(ds[0], ds[1])
    elif md==cgb:
        clf = cgb(verbose=False)
        clf.fit(ds[0], ds[1])
    elif md==efdt:
        print('EFDT')
        clf = md(min_samples_reevaluate=1000)
        clf.fit(ds[0], ds[1])
    else:
        clf = md()
        clf.fit(ds[0], ds[1])
        
    if len(ds) == 2:
        # res = clf.score(ds[0], ds[1])
        pred = clf.predict(ds[0])
        res = accuracy_score(ds[1], pred)
        # res = roc_auc_score(ds[1], pred)
        print(f'{res*100:.3f}')
    else:
        pred = clf.predict(ds[2])
        # res = clf.score(ds[2], ds[3])
        res = accuracy_score(ds[3], pred)
        # res = roc_auc_score(ds[3], pred)
        print(f'{res*100:.3f}')
    return res

def evalate(md, ds):
    ds = read_libsvm(ds)
    print('Vanilla:', end=' ')
    ori = test_md_ds(md, ds)

    nf = len(ds[0][0])
    for i in range(nf):
        trans_phase(ds, i)
    trans_ope(ds)

    print('ENC:', end=' ')
    now = test_md_ds(md, ds)
    # print(f'error: {100*(now-ori):.3f}')

def evalate_all(md, dss=['iris', 'mushrooms', 'cod-rna', 'covtype', 'sensorless', 'heart', 'dna', 'madelon']):
    for ds in dss:
        print(ds)
        evalate(md, ds)
        print('')