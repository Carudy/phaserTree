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
from scipy.interpolate import PchipInterpolator
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import math


DATA_PATH = Path('D://dataset')


def uci_preprocess(dname, original_train, original_test):
    num_train = len(original_train)
    original = pd.concat([original_train, original_test])

    if dname == 'adult':
        labels = original.iloc[:, -1]
        labels = labels.replace('<=50K', 0).replace('>50K', 1)
        labels = labels.replace('<=50K.', 0).replace('>50K.', 1)
        labels = labels.to_numpy()
        del original[3]
        del original[14]

    else:
        labels = original.iloc[:, -1]
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        del original[original.columns[-1]]

    data = pd.get_dummies(original)
    train_data = data[:num_train]
    train_labels = labels[:num_train]
    test_data = data[num_train:]
    test_labels = labels[num_train:]

    return train_data.to_numpy(), train_labels, test_data.to_numpy(), test_labels


def read_uci(dname):
    dir = DATA_PATH / dname
    original_train = pd.read_csv(str(
        dir / f'{dname}.data'), sep=r'\s*,\s*|;', engine='python', na_values="?", header=None)

    try:
        original_test = pd.read_csv(str(
            dir / f'{dname}.test'), sep=r'\s*,\s*|;', engine='python', na_values="?", skiprows=1, header=None)
    except:
        n = len(original_train)
        nt = int(n*0.8)
        original_test = original_train[nt:]
        original_train = original_train[:nt]

    return uci_preprocess(dname, original_train, original_test)


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


def auto_read_dataset(dname):
    if dname in ['adult', 'bank', 'credit', 'higgs', 'nomao']:
        return read_uci(dname)
    return read_libsvm(dname)


class OPE:
    def __init__(self):
        self.a = random.randint(1, 1 << 2)
        self.b = random.randint(1, 1 << 2)
        self.c = random.randint(1, 1 << 2)

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


class SplinePhaser:
    def __init__(self, arr, n):
        if len(arr) <= 1:
            self.spline = lambda x: x
            return
        mi = min(arr)
        ma = max(arr)
        if not math.isfinite(mi) or not math.isfinite(ma):
            self.spline = lambda x: x
            return 
        
        r = ma - mi
        a = random.random() * 0.4 + 0.1
        margin = a * r
        mi, ma = mi - margin, ma + margin

        p = (ma-mi) / n
        l = mi
        sm = 1e-6
        xs = []
        ys = []
        pieces = []
        for i in range(n):
            r = l + p
            pieces.append((l, r))
            l = r
        to_pieces = pieces[:]
        random.seed(n)
        while to_pieces == pieces:
            random.shuffle(to_pieces)
        for i in range(n):
            xs.append(pieces[i][0] + sm)
            xs.append(pieces[i][1])
            ys.append(to_pieces[i][0])
            ys.append(to_pieces[i][1])
        self.spline = PchipInterpolator(xs, ys)
        self.pieces = pieces[:]

    def __call__(self, x):
        return self.spline(x)


def trans_phase_single(ds, i):
    nf = len(ds[0][0])
    A = [x[i] for x in ds[0]]
    pa = SplinePhaser(list(set(A)), 2)
    for x in ds[0]:
        x[i] = pa(x[i])
    for x in ds[2]:
        x[i] = pa(x[i])


def trans_phase(ds, n=2):
    nf = len(ds[0][0])
    for i in range(nf):
        A = [x[i] for x in ds[0]]
        pa = SplinePhaser(list(set(A)), n=n)
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


def test_md_ds(md, ds, shit=True):
    if md == xgb:
        clf = xgb.XGBRegressor(tree_method="gpu_hist")
        clf.fit(ds[0], ds[1])
    elif md == lgb:
        clf = lgb()
        clf.fit(ds[0], ds[1])
    elif md == cgb:
        clf = cgb(verbose=False)
        clf.fit(ds[0], ds[1])
    elif md == efdt:
        print('EFDT')
        clf = md(min_samples_reevaluate=1000)
        clf.fit(ds[0], ds[1])
    else:
        clf = md()
        clf.fit(ds[0], ds[1])

    if shit:
        print(f'(depth: {clf.get_depth()})', end=' ')

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


def evalate(md, ds, only_enc=False, n_piece=2):
    ds = auto_read_dataset(ds)
    if not only_enc:
        print('Vanilla:', end=' ')
        ori = test_md_ds(md, ds)
    else:
        ori = 0

    trans_phase(ds, n=n_piece)

    print('ENC:', end=' ')
    now = test_md_ds(md, ds)
    return (ori, now)
    # print(f'error: {100*(now-ori):.3f}')


def evalate_all(md, dss=['iris', 'mushrooms', 'cod-rna', 'covtype', 'sensorless', 'heart', 'dna', 'madelon'], n_piece=2):
    for ds in dss:
        print(ds)
        evalate(md, ds, n_piece=n_piece)
        print('')
