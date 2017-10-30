import argparse
import os
import codecs
import sys
import StringIO
import second_layer
from tree import read_conll
import tree
#from features import convert_toNumbers
import json
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import pickle


def train(args, devel_split=0):
    """
    main() to launch everything
    """

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    ccprop=second_layer.ConjPropagation()
    
    count=0
    print >> sys.stderr, "collecting training data"
    examples=[]
    labels=[]
    for comments,sent in read_conll(args.input):
        t=tree.Tree(sent,count+1)
        e,l=ccprop.learn(t)
        examples+=e
        labels+=l
        count+=1
    print >> sys.stderr, "sentences:", count, "training examples:", len(examples)

    devel_examples=[]
    devel_labels=[]
    count=0
    if args.devel_data:
        for comments,sent in read_conll(args.devel_data):
            t=tree.Tree(sent,count+1)
            e,l=ccprop.learn(t)
            devel_examples+=e
            devel_labels+=l
            count+=1
        print >> sys.stderr, "sentences:", count, "devel examples:", len(devel_examples)

    if len(devel_examples)==0 and devel_split > 0:
        
        print >> sys.stderr, "splitting training data into train and devel"
        train_examples, train_labels = examples[:len(examples)-devel_split],labels[:len(labels)-devel_split]
        devel_examples, devel_labels = examples[len(examples)-devel_split:],labels[len(labels)-devel_split:]
        print >> sys.stderr, "train examples:",len(train_examples),"devel examples:",len(devel_examples)
    else:
        train_examples, train_labels = examples, labels

    vectorizer=DictVectorizer()
    train_examples=vectorizer.fit_transform(train_examples)
    devel_examples=vectorizer.transform(devel_examples)
    
    c_values=[2**i for i in range(-5, 15)]
    print >> sys.stderr, "c values:", " ".join(str(c) for c in c_values)
    classifiers = [LinearSVC(C=c, random_state=1) for c in c_values]
    evaluation=[]
    for classifier in classifiers:
        print >> sys.stderr, "training classifier with C value", classifier.C
        classifier.fit(train_examples, train_labels)
        predictions=classifier.predict(devel_examples)
        f_score=f1_score(devel_labels,predictions)
        print >> sys.stderr, "f-score:",f_score
        evaluation.append((classifier, f_score))
    
    evaluation.sort(key=lambda x: x[1], reverse=True)
    best_classifier, best_f_score = evaluation[0]
    print >> sys.stderr, "best classifier with c value", best_classifier.C
    print >> sys.stderr, "f_score", best_f_score

    # save vectorizer and model
    with open(args.output+"/ccprop.vectorizer","wb") as f:
        pickle.dump(vectorizer,f)
    with open(args.output+"/ccprop.model","wb") as f:
        pickle.dump(best_classifier,f)
    
    print >> sys.stderr, "...done. Models saved into ", args.output


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    g=parser.add_argument_group("Input/Output")
    g.add_argument('input', nargs='?', help='Training file name, or nothing for training on stdin')
    g.add_argument('-o', '--output', required=True, help='Name of the output model.')
    g.add_argument('--devel_data', type=str, help='File where to read development data.')
    g.add_argument('--devel_split', type=int, default=0, help='How many sentences from training data is used for devel data (if not devel_data given)?')
    args = parser.parse_args()
    
    if args.devel_data is None and args.devel_split==0:
        print >> sys.stderr, "We need development data, use either --devel_data (file) or --devel_split (split train into train+devel)"

    train(args, devel_split=args.devel_split)
