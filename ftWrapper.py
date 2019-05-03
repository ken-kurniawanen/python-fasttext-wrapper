"""
===fastText Python Wrapper===

    This module calls Facebook fastText shell from your beloved Python.
It reproduces fastText function by calling shell command in Python. Also handled input and output conversion.

PS : Use the fastText default format __label__{label}
"""

import os.path
from subprocess import run
import subprocess

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore') 

# specify fastText directory
Path = '/jet/prs/workspace/fastText-0.2.0/'


def train(X,y, pretrained = None, dim=300 ,Ngrams=1,epochs=5, minn=3,maxn=6, model="model"):
    """
    Equivalent to ./fasttext supervised
    
    produce model.bin in fastText directory for predict purpose

    X = Series or list containing text
    y = Series or list containing label

    Option :
    -pretrainedVectors  specify path of pretrained word vectors for supervised learning []
    -wordNgrams         max length of word ngram [1]
    -epoch              number of epochs [5]
    -minn               min length of char ngram [3]
    -maxn               max length of char ngram [6]
    -model              specify name for model .bin and .vec if you don't want to overwrite the default model.bin
    """

    cmd = Path + "fasttext supervised -input " + Path + "train__Data.txt" + " -output " + Path + "model" + " -dim " + str(dim) + " -minn " + str(minn) + " -maxn " + str(maxn) +" -wordNgrams " + str(Ngrams) + " -epoch " + str(epochs)

    if pretrained:
        cmd = cmd + " -pretrainedVectors " + pretrained

    # input handling
    if os.path.isfile(Path + "train__Data.txt"):
        os.remove(Path + "train__Data.txt", dir_fd=None)

    with open(Path + 'train__Data.txt', 'a') as f:
        for i,x in enumerate(X):
            f.write('__label__' + str(y[i]) + ' ' + x)
            f.write('\n')       
    
    # error checking, pass STDERR or STDOUT to Python
    try:
        out = subprocess.run(cmd, stderr=subprocess.PIPE, shell=True, check=True)
        #print(cmd)
        #print(out.stderr.decode('utf-8'))
    except Exception:
        result = subprocess.run(cmd, stderr=subprocess.PIPE, shell=True)
        print(cmd)
        print(result.stderr.decode('utf-8'))

    
def score(X,y):
    """
    Equivalent of fastText ./fasttext test

    X = Series or list containing text
    y = Series or list containing label
    """
    # input handling
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
    """
    Equivalent of  ./fasttext predict-prob

    X = Series or list containing text to predict
    """
    if os.path.isfile(Path + "predict__Data.txt"):
        os.remove(Path + "predict__Data.txt", dir_fd=None)
    with open(Path + 'predict__Data.txt', 'a') as f:
        for x in X:
            f.write(str(x))
            f.write('\n')

    run(Path + "fasttext predict-prob " + Path + "model.bin " + Path + "predict__Data.txt > " + Path + "outputPredictions.txt",shell=True)
    
    # read the predictions and output as list
    with open(Path + 'outputPredictions.txt') as f:
        outputPredictions = f.readlines()
    
    return outputPredictions

def predict(X, model="model.bin"):
    """
    Equivalent of  ./fasttext predict
    Return list of prediction per line

    X = Series or list containing text to predict

    Option :
    -model              specify name for model .bin and .vec in fastText directory
    """

    # input handling
    if os.path.isfile(Path + "predict__Data.txt"):
        os.remove(Path + "predict__Data.txt", dir_fd=None)
    with open(Path + 'predict__Data.txt', 'a') as f:
        for x in X:
            f.write(str(x))
            f.write('\n')
            
    run(Path + "fasttext predict " + Path + model + " " + Path + "predict__Data.txt > " + Path + "outputPredictions.txt", shell=True)
    
    # read the prediction and output as list
    with open(Path + 'outputPredictions.txt') as f:
        outputPredictions = f.readlines()

    # optional adjustment to remove _label_
    for idx,i in enumerate(outputPredictions):
        outputPredictions[idx] = ''.join(filter(lambda x: x.isdigit(), i))

    return outputPredictions

def cross_val_score(data, pretrained = None, dim=300 ,Ngrams=1,epochs=5, minn=3,maxn=6,k=5):
    """
    Run cross validation based on fastText model and sklearn style data.
    Return list of evaluation metric.
    Output evaluation metric list's mean and confusion matrix of last fold.

    data = DataFrame containing "text" column and "label" column

    Option:
    Option :
    -pretrainedVectors  specify path of pretrained word vectors for supervised learning []
    -wordNgrams         max length of word ngram [1]
    -epoch              number of epochs [5]
    -minn               min length of char ngram [3]
    -maxn               max length of char ngram [6]
    -k                  number of fold [5]
    """

    acc_list = []
    precision_list = []
    f1_list = []
    
    for fold in range(1,k+1):
        
        test_ = data.loc[len(data)//k*(fold-1):len(data)//k*fold-1]
        train_ = data[~data.index.isin(test_.index)]
        
        train(train_['text'].tolist(),train_['label'].tolist(), dim=dim, pretrained=pretrained, minn=minn ,maxn=maxn, epochs = epochs)
        pred = predict(test_['text'].tolist())

        acc_list.append(accuracy_score(pred, test_['label'].tolist()))
        precision_list.append(precision_score(pred, test_['label'].tolist(), average='weighted'))
        f1_list.append(f1_score(pred, test_['label'].tolist(), average='weighted'))
        conf_matrix = confusion_matrix(pred, test_['label'].tolist())

    # print("Accuracy : %0.3f (+/- %0.2f)" % (np.mean(acc_list), np.std(acc_list) * 2))
    # print("Precision : %0.3f (+/- %0.2f)" % (np.mean(precision_list), np.std(precision_list) * 2))
    # print("F1-score : %0.3f (+/- %0.2f)" % (np.mean(f1_list), np.std(f1_list) * 2))
    # print(conf_matrix)
    
    return acc_list,precision_list,f1_list

def simulation(data, pretrained = None, dim=300 ,Ngrams=1,epochs=5, minn=3,maxn=6, metric=["accuracy","precision","f1"]):
    
    train_size_list = np.linspace(0.9,0.1,9)

    text = list(data['text'])
    target = list(data['label'])

    for idx, train_size in enumerate(train_size_list):
        print("train size : ", train_size, end=' ')

        text_train,text_validation,target_train,target_validation = train_test_split(text,target, test_size=1-train_size, random_state=1)
        
        train(X=text_train, y=target_train, dim=dim, pretrained=pretrained, minn=minn, maxn=maxn, epochs=epochs)

        if "accuracy" in metric:
            print('accuracy : %0.3f'%accuracy_score(predict(text_validation), target_validation),end=' ')
        if "precision" in metric:
            print('precision : %0.3f'%precision_score(predict(text_validation),target_validation, average="weighted"),end=' ')
        if "f1" in metric:
            print('f1 score : %0.3f'%f1_score(predict(text_validation),target_validation, average="weighted"))
        if "confusion_matrix" in metric:
            print(confusion_matrix(predict(text_validation),target_validation))

    return 

def grid_search(data,pretrains=[None,"wiki.id.vec"], epochs=[5,15,50], subwords=["no","yes"]):
    print('[pretrain, epoch, subword]-> accuracy = x, f1 = x')

    for pretrain in pretrains:
        for epoch in epochs:
            for subword in subwords:
                if 'yes' in subwords:
                    minn = 3
                    maxn = 6
                else:
                    minn = 0
                    maxn = 0

                acc_list,_,f1_list = cross_val_score(data,pretrain,dim=300,Ngrams=1,epochs=epoch,minn=minn,maxn=maxn,k=5)
                print('[',str(pretrain),',',str(epoch),',',str(subword),']', end=' ')
                print('a : %0.3f (+/- %0.2f)' % (np.mean(acc_list), np.std(acc_list) * 2), end=' ')
                print('f1 : %0.3f (+/- %0.2f)' % (np.mean(f1_list), np.std(f1_list) * 2))

    return