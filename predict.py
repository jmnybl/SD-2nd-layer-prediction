import argparse
import os
import sys
import second_layer
from tree import read_conll
import tree
#from features import convert_toNumbers
import json
import time
import pickle

SCRIPTDIR=os.path.dirname(os.path.abspath(__file__))

def predict(args):
    """
    main() to launch everything
    """

    if not os.path.exists(args.model):
        print >> sys.stderr, "no model", args.model
        sys.exit(1)

    print >> sys.stderr, "loading models"
#    relmodel=second_layer.Model(args.model[0],u"rel")    
#    rel=second_layer.Relativizers(relmodel)
#    ccmodel=second_layer.Model(args.model[0],u"ccprop")

    with open(args.model+"/ccprop.vectorizer","rb") as f:
        vectorizer=pickle.load(f)
    with open(args.model+"/ccprop.model","rb") as f:
        classifier=pickle.load(f)

    ccprop=second_layer.ConjPropagation(model=classifier,vectorizer=vectorizer)
    xs=second_layer.Xsubjects()
    
    print >> sys.stderr, "predicting"
    count=0
    for comments,sent in read_conll(args.input):
        t=tree.Tree(sent,count+1)
#        rel.predict(t)
        xs.predict(t)
        ccprop.predict(t)
        xs.predict(t)
        for comm in comments:
            print >> sys.stdout, comm.encode(u"utf-8")
        if args.no_conllu:
            t.tree_to_conll()
        else:
            t.to_conllu()
        count+=1
        if count%500==0:
            print >> sys.stderr, count
        
    print >> sys.stderr, "sentences:",count
    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Runs the second layer prediction. Outputs to stdout.')
    parser.add_argument('-m','--model', help='Name of the model directory.')
    parser.add_argument('--no-conllu', dest='no_conllu', default=False, action="store_true", help='Do not output conllu, use conll09. Default: %(default)s')
    parser.add_argument('-i','--input', help='Input file name, or nothing for stdin')
    args = parser.parse_args()
    
    predict(args)
