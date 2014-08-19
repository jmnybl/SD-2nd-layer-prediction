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

SCRIPTDIR=os.path.dirname(os.path.abspath(__file__))


def train(args):
    """
    main() to launch everything
    """

    if not args.no_ccprop:
        if not os.path.exists(os.path.join(SCRIPTDIR,args.output,u"ccprop")):
            os.makedirs(os.path.join(SCRIPTDIR,args.output,u"ccprop"))
        cc_trainf=codecs.open(os.path.join(SCRIPTDIR,args.output,u"ccprop","train.txt"),"wt",u"utf-8")
        ccprop=second_layer.ConjPropagation()
    else:
        ccprop=None
        
    if not args.no_rel:
        if not os.path.exists(os.path.join(SCRIPTDIR,args.output,u"rel")):
            os.makedirs(os.path.join(SCRIPTDIR,args.output,u"rel"))
        rel_trainf=codecs.open(os.path.join(SCRIPTDIR,args.output,u"rel","train.txt"),"wt",u"utf-8")
        rel=second_layer.Relativizers()
    else:
        rel=None
    
    count=0
    print >> sys.stderr, "collecting training data"
    for sent in read_conll(args.input):
        t=tree.Tree(sent)
        if rel is not None:
            rel.learn(t,rel_trainf)
        if ccprop is not None:
            ccprop.learn(t,cc_trainf)
        count+=1
    print >> sys.stderr, "sentences:",count

    print >> sys.stderr, "converting training files"
    if not args.no_ccprop:
        cc_trainf.close()
        convert_toNumbers(False,u"ccprop",args.output)
        
        
    if not args.no_rel:
        rel_trainf.close()
        convert_toNumbers(False,u"rel",args.output)

    
    ## TODO svm


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Trains the parser in a multi-core setting.')
    g=parser.add_argument_group("Input/Output")
    g.add_argument('input', nargs='?', help='Training file name, or nothing for training on stdin')
    g.add_argument('-o', '--output', required=True, help='Name of the output model.')
    g=parser.add_argument_group("Training config")
    g.add_argument('--no_ccprop', required=False, dest='no_ccprop', action="store_true", default=False, help='Do not train conjunct propagation model. (default %(default)d)')
    g.add_argument('--no_rel', required=False, dest='no_rel', action="store_true", default=False, help='Do not train relativizer prediction model. (default %(default)d)')

#    g.add_argument('-p', '--processes', type=int, default=4, help='How many training workers to run? (default %(default)d)')
#    g.add_argument('--max_sent', type=int, default=0, help='How many sentences to read from the input? 0 for all.  (default %(default)d)')
#    g=parser.add_argument_group("Training algorithm choices")
#    g.add_argument('-i', '--iterations', type=int, default=10, help='How many iterations to run? If you want more than one, you must give the input as a file. (default %(default)d)')
#    g.add_argument('--dim', type=int, default=5000000, help='Dimensionality of the trained vector. (default %(default)d)')
#    g.add_argument('--beam_size', type=int, default=40, help='Size of the beam. (default %(default)d)')
    args = parser.parse_args()
    
    train(args)
