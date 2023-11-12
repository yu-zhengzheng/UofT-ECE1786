import pandas as pd
import sklearn.model_selection
import sklearn.utils
import numpy as np
def gen_files():
    #generate train val test data tsv files
    df = pd.read_csv("data\data.tsv", sep="\t")

    train0, test0 = sklearn.model_selection.train_test_split(df[0:5000], test_size=0.2, random_state=41)
    overfit0=train0[0:25]
    train0, val0 = sklearn.model_selection.train_test_split(train0, test_size=0.2, random_state=41)
    train1, test1 = sklearn.model_selection.train_test_split(df[5000:10000], test_size=0.2, random_state=41)
    overfit1=train1[0:25]
    train1, val1 = sklearn.model_selection.train_test_split(train1, test_size=0.2, random_state=41)

    train=sklearn.utils.shuffle(pd.concat([train0,train1]))
    val=sklearn.utils.shuffle(pd.concat([val0,val1]))
    test=sklearn.utils.shuffle(pd.concat([test0,test1]))
    overfit=sklearn.utils.shuffle(pd.concat([overfit0,overfit1]))

    train.to_csv('data\\train.tsv', sep="\t")
    val.to_csv('data\\validation.tsv', sep="\t")
    test.to_csv('data\\test.tsv', sep="\t")
    overfit.to_csv('data\\overfit.tsv', sep="\t")

    print("amount/proportion of objective samples in train:",np.sum(train["label"]),np.sum(train["label"])/len(train))
    print("amount/proportion of objective samples in val:",np.sum(val["label"]),np.sum(val["label"])/len(val))
    print("amount/proportion of objective samples in test:",np.sum(test["label"]),np.sum(test["label"])/len(test))
    print("amount/proportion of objective samples in overfit:",np.sum(overfit["label"]),np.sum(overfit["label"])/len(overfit))
gen_files()
#%%