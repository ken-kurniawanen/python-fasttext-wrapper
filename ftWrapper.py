"""
===fastText Python Wrapper===

    This module calls Facebook fastText C++ library from your beloved Python.
It reproduces fastText function by calling shell command in Python. Also handled input and output conversion.

PS : Use the fastText default format __label__{label}
"""

import os.path
from subprocess import run
import subprocess

# Specify fastText directory
Path = '/jet/prs/workspace/fastText-0.2.0/'


def train(X,y, pretrained = None, dim=100 ,Ngrams=1,epochs=5, minn=3,maxn=6, model="model"):
    """
    Equivalent to ./fasttext supervised

    Option :
    -pretrainedVectors  pretrained word vectors for supervised learning []
    -wordNgrams         max length of word ngram [1]
    -epoch              number of epochs [5]
    -minn               min length of char ngram [3]
    -maxn               max length of char ngram [6]
    -model              specify your path and name for model .bin and .vec if you don't want to overwrite the default model.bin
    """

    cmd = Path + "fasttext supervised -input " + Path + "train__Data.txt" + " -output " + Path + "model" + " -dim " + str(dim) + " -minn " + str(minn) + " -maxn " + str(maxn) +" -wordNgrams " + str(Ngrams) + " -epoch " + str(epochs)

    if pretrained:
        cmd = cmd + " -pretrainedVectors " + pretrained

    if os.path.isfile(Path + "train__Data.txt"):
        os.remove(Path + "train__Data.txt", dir_fd=None)

    with open(Path + 'train__Data.txt', 'a') as f:
        for i,x in enumerate(X):
            f.write('__label__' + str(y[i]) + ' ' + x)
            f.write('\n')       
    
    try:
        out = subprocess.run(cmd, stderr=subprocess.PIPE, shell=True, check=True)
        print(cmd)
        print(out.stderr.decode('utf-8'))
    except Exception:
        result = subprocess.run(cmd, stderr=subprocess.PIPE, shell=True)
        print(cmd)
        print(result.stderr.decode('utf-8'))

    
def score(X,y):
    if os.path.isfile(Path + "test__Data.txt"):
        os.remove(Path + "test__Data.txt", dir_fd=None)
    with open(Path + 'test__Data.txt', 'a') as f:
        for i,x in enumerate(X):
            f.write('__label__' + str(y[i]) + ' ' + x)
            f.write('\n')

    run(Path + "fasttext test " + Path + "model.bin " + Path + "test__Data.txt > " + Path + "outputScores.txt", shell=True)
    
    with open(Path + 'outputScores.txt') as f:
        for e in f.readlines():
            print(e)
        
def predict_proba(X):
    if os.path.isfile(Path + "predict__Data.txt"):
        os.remove(Path + "predict__Data.txt", dir_fd=None)
    with open(Path + 'predict__Data.txt', 'a') as f:
        for x in X:
            f.write(str(x))
            f.write('\n')

    run(Path + "fasttext predict-prob " + Path + "model.bin " + Path + "predict__Data.txt > " + Path + "outputPredictions.txt",shell=True)
    ##read the predictions and print them
    with open(Path + 'outputPredictions.txt') as f:
        outputPredictions = f.readlines()
    
    return outputPredictions

def predict(X):
    if os.path.isfile(Path + "predict__Data.txt"):
        os.remove(Path + "predict__Data.txt", dir_fd=None)

    with open(Path + 'predict__Data.txt', 'a') as f:
        for x in X:
            f.write(str(x))
            f.write('\n')
            
    run(Path + "fasttext predict " + Path + "model.bin " + Path + "predict__Data.txt > " + Path + "outputPredictions.txt", shell=True)
    
    with open(Path + 'outputPredictions.txt') as f:
        outputPredictions = f.readlines()

    for idx,i in enumerate(outputPredictions):
        outputPredictions[idx] = ''.join(filter(lambda x: x.isdigit(), i))

    return outputPredictions

def cross_val_score():
    acc_list = []
    precision_list = []
    f1_list = []
    
    for fold in range(1,k+1):
        
        test = data.loc[len(data)//k*(fold-1):len(data)//k*fold-1]
        train = data[~data.index.isin(test.index)]
        
        ftWrapper.train(train['text'].tolist(),train['label'].tolist(), dim=dim, pretrained=pretrained, minn=minn ,maxn=maxn, epochs = epochs)
        pred = ftWrapper.predict(test['text'].tolist())

        acc_list.append(accuracy_score(pred, test['label'].tolist()))
        precision_list.append(precision_score(pred, test['label'].tolist(), average='weighted'))
        f1_list.append(f1_score(pred, test['label'].tolist(), average='weighted'))
        conf_matrix = confusion_matrix(pred, test['label'].tolist())

    print("Accuracy : %0.3f (+/- %0.2f)" % (np.mean(acc_list), np.std(acc_list) * 2))
    print("Precision : %0.3f (+/- %0.2f)" % (np.mean(precision_list), np.std(precision_list) * 2))
    print("F1-score : %0.3f (+/- %0.2f)" % (np.mean(f1_list), np.std(f1_list) * 2))
    print(conf_matrix)
    
    return acc_list,precision_list,f1_list

