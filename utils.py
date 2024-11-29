#!/usr/bin/env python


import os
import re
import sys
import h5py
import time
import string
import contextlib
import multiprocessing
import pandas as pd
import numpy as np
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf


identifier = '(([a-zA-Z]_)?([a-zA-Z0-9_]*))'
replacement_field = '{' + identifier + '}'


def reset():
    ops.reset_default_graph()
    tf.set_random_seed(19)


def load_phenotypes(pheno_path):
    pheno = pd.read_csv(pheno_path)
    
    pheno = pheno[pheno['FILE_ID'] != 'no_filename']
    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: int(v)-1)
    pheno['SITE_ID'] = pheno['SITE_ID'].apply(lambda v: re.sub('_[0-9]', '', v))
    pheno['MEAN_FD'] = pheno['func_mean_fd']
    pheno['SUB_IN_SMP'] = pheno['SUB_IN_SMP'].apply(lambda v: v == 1)
    pheno["STRAT"] = pheno[["SITE_ID", "DX_GROUP"]].apply(lambda x: "_".join([str(s) for s in x]), axis=1)
    pheno["AGE"] = pheno['AGE_AT_SCAN']
    pheno['SEX'] = pheno['SEX']
    pheno["FIQ"] = pheno['FIQ'].fillna(pheno['FIQ'].mean())
    pheno["VIQ"] = pheno['VIQ'].fillna(pheno['VIQ'].mean())
    pheno["PIQ"] = pheno['PIQ'].fillna(pheno['PIQ'].mean())
    #pheno['HANDEDNESS_SCORES'] = pheno['HANDEDNESS_SCORES'].fillna(method='bfill')

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID', 'DX_GROUP', 'SEX', 'SITE_ID', 'MEAN_FD', 'SUB_IN_SMP', 'STRAT','AGE','FIQ','VIQ','PIQ']]

def hdf5_handler(filename, mode="r"):
    h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename, fapl=propfaid)) as fid:
        f = h5py.File(fid, mode)
        return f

def load_fold(patients, experiment, fold):

    derivative = experiment.attrs["derivative"]

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)
    #print(pheno['FILE_ID'])

    X_train = []
    y_train = []

    for pid in experiment[fold]["train"]:
        
        p=pheno[pheno['FILE_ID']==pid.decode(encoding = 'UTF-8')]
        x=np.array(patients[pid][derivative])
        x = np.append(x, int(p['SEX'].values))         # Sex as integer (e.g., 1 for male, 2 for female)
        x = np.append(x, float(p['AGE'].values))       # Age
        x = np.append(x, float(p['FIQ'].values))       # VIQ (verbal IQ)
        x = np.append(x, float(p['VIQ'].values))       # VIQ (verbal IQ)
        x = np.append(x, float(p['PIQ'].values))       # PIQ (performance IQ)
        #x = np.append(x, int(p['HANDEDNESS_SCORES'].values))  # Handedness

        X_train.append(x)
        y_train.append(patients[pid].attrs["y"])

    X_valid = []
    y_valid = []
    for pid in experiment[fold]["valid"]:

        p=pheno[pheno['FILE_ID']==pid.decode(encoding = 'UTF-8')]
        x=np.array(patients[pid][derivative])
        x=np.append(x,int(p['SEX'].values))
        x=np.append(x,float(p['AGE'].values))
        x = np.append(x, float(p['FIQ'].values))       # VIQ (verbal IQ)
        x = np.append(x, float(p['VIQ'].values))       # VIQ (verbal IQ)
        x = np.append(x, float(p['PIQ'].values))       # PIQ (performance IQ)
        #x = np.append(x, int(p['HANDEDNESS_SCORES'].values))  # Handedness

        X_valid.append(x)
        y_valid.append(patients[pid].attrs["y"])

    X_test = []
    y_test = []
    for pid in experiment[fold]["test"]:

        p=pheno[pheno['FILE_ID']==pid.decode(encoding = 'UTF-8')]
        x=np.array(patients[pid][derivative])
        x=np.append(x,int(p['SEX'].values))
        x=np.append(x,float(p['AGE'].values))
        x = np.append(x, float(p['FIQ'].values))       # VIQ (verbal IQ)
        x = np.append(x, float(p['VIQ'].values))       # VIQ (verbal IQ)
        x = np.append(x, float(p['PIQ'].values))       # PIQ (performance IQ)
        #x = np.append(x, int(p['HANDEDNESS_SCORES'].values))  # Handedness

        X_test.append(x)
        y_test.append(patients[pid].attrs["y"])


    return np.array(X_train), y_train, \
           np.array(X_valid), y_valid, \
           np.array(X_test), y_test

class SafeFormat(dict):

    def __missing__(self, key):
        return "{" + key + "}"

    def __getitem__(self, key):
        if key not in self:
            return self.__missing__(key)
        return dict.__getitem__(self, key)

def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def format_config(s, *d):
    dd = merge_dicts(*d)
    return string.Formatter().vformat(s, [], SafeFormat(dd))


def elapsed_time(tstart):
    tnow = time.time()
    total = tnow - tstart
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def run_progress(callable_func, items, message=None, jobs=1):

    results = []

    print ('Starting pool of %d jobs' % jobs)

    current = 0
    total = len(items)

    if jobs == 1:
        results = []
        for item in items:
            results.append(callable_func(item))
            current = len(results)
            if message is not None:
                args = {'current': current, 'total': total}
                sys.stdout.write("\r" + message.format(**args))
                sys.stdout.flush()

    # Or allocate a pool for multithreading
    else:
        pool = multiprocessing.Pool(processes=jobs)
        for item in items:
            pool.apply_async(callable_func, args=(item,), callback=results.append)

        while current < total:
            current = len(results)
            if message is not None:
                args = {'current': current, 'total': total}
                sys.stdout.write("\r" + message.format(**args))
                sys.stdout.flush()
            time.sleep(0.5)

        pool.close()
        pool.join()

    return results


def root():
    return os.path.dirname(os.path.realpath(__file__))


def to_softmax(n_classes, classe):
    sm = [0.0] * n_classes
    sm[int(classe)] = 1.0
    return sm


def load_ae_encoder(input_size, code_size, model_path):
    model = ae(input_size, code_size)
    init = tf.global_variables_initializer()
    try:
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver(model["params"], write_version= tf.train.SaverDef.V2)
            if os.path.isfile(model_path):
                print ("Restoring", model_path)
                saver.restore(sess, model_path)
            params = sess.run(model["params"])
            return {"W_enc": params["W_enc"], "b_enc": params["b_enc"]}
    finally:
        reset()


def sparsity_penalty(x, p, coeff):
    p_hat = tf.reduce_mean(tf.abs(x), 0)
    kl = p * tf.log(p / p_hat) + \
        (1 - p) * tf.log((1 - p) / (1 - p_hat))
    return coeff * tf.reduce_sum(kl)