import argparse
import os
import codecs
import sys
import StringIO
import second_layer
from tree import read_conll
import tree
from features import convert_toNumbers
import json
import time


def train(args):
    """
    main() to launch everything
    """

    if not os.path.exists(os.path.join(args.output,u"ccprop")):
        os.makedirs(os.path.join(args.output,u"ccprop"))
    cc_trainf=codecs.open(os.path.join(args.output,u"ccprop","train.txt"),"wt",u"utf-8")
    ccprop=second_layer.ConjPropagation()
    
    count=0
    print >> sys.stderr, "collecting training data"
    for comments,sent in read_conll(args.input):
        t=tree.Tree(sent,count+1)
        ccprop.learn(t,cc_trainf)
        count+=1
    print >> sys.stderr, "sentences:",count

    print >> sys.stderr, "converting training files"
    cc_trainf.close()
    convert_toNumbers(False,u"ccprop",args.output)
    
    print >> sys.stderr, "...done. Now train svm model."
    ## TODO svm


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    g=parser.add_argument_group("Input/Output")
    g.add_argument('input', nargs='?', help='Training file name, or nothing for training on stdin')
    g.add_argument('-o', '--output', required=True, help='Name of the output model.')
    args = parser.parse_args()
    
    train(args)
