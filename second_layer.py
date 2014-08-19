#-*- coding: utf-8 -*-


import time

import codecs
import glob
import sys
import os
import cPickle
import math
import subprocess # needed for pigz
import gzip
from collections import defaultdict,namedtuple

from tree import Tree,Dep
from features import JumpFeatures,RelFeatures
import numpy as np


JUMPCLASSES={u"DontJump":1,u"adpos":2,u"advcl":3,u"advmod":4,u"acomp":5,u"amod":6,u"appos":7,u"aux":8,u"cc":9,u"ccomp":10,u"compar":11,
u"comparator":12,u"complm":13,u"conj":14,u"cop":15,u"csubj":16,u"csubj-cop":17,u"dep":18,u"det":19,u"dobj":20,u"ellipsis":21,u"gobj":22,u"iccomp":23,
u"infmod":24,u"mark":25,u"name":26,u"neg":27,u"nn":28,u"nommod":29,u"nsubj":30,u"nsubj-cop":31,u"num":32,u"number":33,u"parataxis":34,u"partmod":35,u"poss":36,u"preconj":37,u"prt":38,u"quantmod":39,u"rcmod":40,
u"rel":41,u"xcomp":42,u"xsubj":43,u"gsubj":44,u"nommod-own":45,u"intj":46,u"punct":47,u"auxpass":48,u"voc":49,u"xsubj-cop":50,u"rel&nsubj":51,u"rel&dobj":52,u"rel&nommod":53,u"rel&nsubj-cop":54,u"rel&advmod":55,u"rel&advcl":56,u"rel&nommod-own":57,u"rel&partmod":58,u"rel&poss":59,u"rel&xcomp":60} # TODO: create this during training and pickle

RELCLASSES={u"nsubj":1,u"dobj":2,u"nommod":3,u"nsubj-cop":4,u"advmod":5,u"advcl":6,u"nommod-own":7,u"partmod":8,u"poss":9,u"xcomp":10}

class Model(object):

    def __init__(self,modelf,featf,classes):
        self.klasses=classes
        self.vector=None ## readModel() fills these...
        self.base_features=None
        self.fDict=None
        self.number2klass=None
        self.readModel(modelf,featf)

    def readModel(self,modelFile,fFile):
        print >> sys.stderr, "Reading SVM model..."
        f=codecs.open(modelFile,u"rt",u"utf-8")
        self.vector=None
        for line in f:
            line=line.strip()
            if u"number of base features" in line:
                baseFs=int(line.split()[0])
                self.base_features=baseFs
            elif u"highest feature index" in line:
                self.vector=np.zeros(int(line.split()[0]))
            elif u"qid:0" in line:
                cols=line.split()
                del cols[0:2]
                del cols[-1]
                for col in cols:
                    parts=col.split(u":")
                    fNum=int(parts[0])
                    value=float(parts[1])
                    self.vector[fNum]=value
        f.close()
        print >> sys.stderr,"Model:",modelFile
        f=codecs.open(fFile,"rb")
        self.fDict=cPickle.load(f)
        f.close()
        print >> sys.stderr,"Feature dictionary:",fFile
#        f=codecs.open(klassFile,"rb")
#        klasses=cPickle.load(f)
#        f.close()
        ## convert from key:ArgName,value:klassNum to key:klassNum,value:ArgName
        self.number2klass={}
        for key,value in self.klasses.iteritems():
            self.number2klass[value]=key

    def predict_one(self,features):
        """ features are in text format, first convert to numbers """
        fnums=[]
        for feature in features:
            feat=self.fDict.get(feature)
            if feat is not None:
                fnums.append((feat,1.0))

        maxscore=None
        klassnum=None
        for i in xrange(1,len(self.klasses)):
            score=0.0
            offset=((i-1)*self.base_features)
            for fnum,value in fnums: # list of feature numbers
                w=1/(math.sqrt(len(fnums)))
                pos=offset+fnum # position in model.vector
                score+=w*self.vector[pos]
                if (maxscore is None) or maxscore<score:
                    maxscore=score
                    klassnum=i
        return klassnum





#CoNLLFormat=namedtuple("CoNLLFormat",["ID","FORM","LEMMA","POS","FEAT","HEAD","DEPREL"])

##Column lists for the various formats
#formats={"conll09":CoNLLFormat(0,1,2,4,6,8,10)}


#def create_tree(sent,conll_format=u"conll09"):
#    """ returns a key:(g,d) value:[type] dictionary"""
#    form=formats[conll_format]
#    base_tree=collection.defaultdict(lambda:[])
#    full_tree=collection.defaultdict(lambda:[])
#    for tok in sent:
#        if tok[form.HEAD]==u"0": continue # skip root
#        govs=tok[form.HEAD].split(u",")
#        dtypes=tok[form.DEPREL].split(u",")
#        dep=int(tok[0])
#        i=0
#        for gov,dtype in zip(govs,dtypes):
#            if i<1:
#                base_tree[(int(gov),dep)].append(dtype)
#            full_tree[(int(gov),dep)].append(dtype)
#            i+=1
#    return base_tree,full_tree

def is_dep(g,d,tree):
    """ Check if there is dependency between g and d, and return list of types if found. """
    types=[]
    for dep in tree.deps:
        if g==dep.gov and d==dep.dep:
            types.append(dep.dtype)
    return types

#Writes the textfile, which has to be later converted to numbers and feed to svm.
def writeData(f,klass,features):
    feat_str=u" ".join(feat for feat in features)
    f.write(klass+u" "+feat_str+u"\n")


class ConjPropagation(object):

    def __init__(self,model=None):
        self.features=JumpFeatures()
        self.model=model
        self.resttypes=set([u"cc",u"conj",u"punct",u"ellipsis"])

    def can_jump(self,dep,tree):
        """ Check whether dep can jump. """
        if dep.dtype in self.resttypes: return False
        for conj in tree.conjs:
            if dep.dep==conj.gov or dep.gov==conj.gov:
                return True
        return False


    def possible_jumps(self,g,d,tree):
        """ g,d is one candidate for propagation, returns a set of 'propagated' (gov,dep) tuples """
        candidates=set()
        for conj in tree.conjs:
            if conj.gov==d: # dependent can move
                candidates.add((g,conj.dep))    
            elif conj.gov==g: # governor can move
                candidates.add((conj.dep,d))        
        return candidates


    # Recursive search for all possible rec jumps
    def gather_all_jumps(self,g,d,tree):
        recs=self.possible_jumps(g,d,tree)
        if len(recs)>0:
            new_set=set()
            for gov,dep in recs:
                new_set|=self.gather_all_jumps(gov,dep,tree)
            recs|=new_set
            return recs
        else:
            return recs


#    def merge_rels(self,tree):
#        merged=[]
#        rels=[]
#        merged_tree={}
#        for g,d,t in tree.iteritems():
#            if u"rel" in t and len(t)>1:
#                assert len(t)<2
#                rels.append((g,d))
#                dtype=u"&".join(ty for ty in t)
#                merged_tree[(g,d)]=[dtype]
#            else:
#                merged_tree[(g,d)]=t
#        return merged_tree

    def learn(self,tree,outfile):
        for dep in tree.deps:
            if dep.flag!=u"CC" and self.can_jump(dep,tree):
                if dep.dtype==u"rel" and len(tree.govs[dep.dep])>1:
                    continue # TODO: do not skip post-processed rel
                new_deps=self.gather_all_jumps(dep.gov,dep.dep,tree)
                for g,d in new_deps:
                    types=is_dep(g,d,tree)
                    if not types:
                        klass=u"no"
                    elif len(types)==2 and u"rel" in types:
                        klass=u"&".join(t for t in types)
                    else:
                        assert len(types)<2
                        klass=types[0]
                    features=self.features.create(dep,g,d,tree)
                    writeData(outfile,klass,features)
                    

    def predict(self,tree):
        """ Jump one conll sentence. """
        new=[]
        for dep in tree.deps:
            if self.can_jump(dep,tree):
                if dep.dtype==u"rel" and len(tree.govs[dep.dep])>1:
                    continue # this is just the rel, we want the secondary function
                new_deps=self.gather_all_jumps(dep.gov,dep.dep,tree)
                for g,d in new_deps:
                    features=self.features.create(dep,g,d,tree) #...should return (name,value) tuples
                    klass=self.model.predict_one(features)      
                    if klass==1: continue
                    klass_str=self.model.number2klass[klass]
                    if u"&" in klass_str: # this is merged rel, split
                        rel,sec=klass_str.split(u"&",1)
                        dependency=Dep(g,d,rel,flag=u"CC")
                        new.append(dependency)
                        dependency=Dep(g,d,sec,flag=u"CC")
                        new.append(dependency)
                    else:
                        dependency=Dep(g,d,klass_str,flag=u"CC")
                        new.append(dependency)
        for dep in new:
            tree.add_dep(dep)



class Relativizers(object):

    def __init__(self,model=None):
        self.features=RelFeatures()
        self.model=model

    def learn(self,tree,outfile):
        for rel in tree.rels:
            features=self.features.create(rel.gov,rel.dep,tree)
            types=is_dep(rel.gov,rel.dep,tree)
            if len(types)<2: continue # post-processed dependency, do not use in training
            klass=types[1]
            writeData(outfile,klass,features)


    def post_process_rel(self,g,d,t,tree):
        # move dependency if iccomp and it does not have this dtype already
        for dep in tree.deps:
            if dep.gov==g and dep.dtype==u"iccomp":
                deps=set()
                for dep2 in tree.childs[dep.dep]: # dependents of iccomp
                    deps.add(dep2.dtype) # collect all dependents of iccomp
                if t not in deps:
                    g=dep.dep
                    break
        return g,d,t

    def predict(self,tree):
        if self.model is None:
            print >> sys.stderr, u"no model found"
            sys.exit(1)
        for rel in tree.rels:
            features=self.features.create(rel.gov,rel.dep,tree)
            klass=self.model.predict_one(features)
            klass_str=self.model.number2klass[klass]
            g,d,t=self.post_process_rel(rel.gov,rel.dep,klass_str,tree)
            dependency=Dep(g,d,t,flag=u"REL")
            #print dependency
            tree.add_dep(dependency)


class Xsubjects(object):

    def decide_type(self,token,tree):
        """ token: gov of new dependency (and dep of xcomp also) """
        for dep in tree.childs[token]:
            if dep.dtype==u"cop":
                return u"xsubj-cop"
        if token.pos==u"Adv" or token.pos==u"N" or token.pos==u"A":
            return u"xsubj-cop"
        return u"xsubj"


    def predict(self,tree):
        new=[]
        for dep in tree.deps:
            if dep.dtype==u"xcomp" and dep.gov in tree.subjs:
                for subj in tree.subjs[dep.gov]:
                    types=is_dep(dep.dep,subj.dep,tree)
                    if (u"xsubj" in types) or u"xsubj-cop" in types: continue # this is because we run xsubj part twice!
                    for dep2 in tree.childs[dep.dep]: # xcomp chain
                        if dep2.dtype==u"xcomp":
                            types=is_dep(dep2.gov,subj.dep,tree)
                            if (u"xsubj" in types or u"xsubj-cop" in types): continue
                            dtype=self.decide_type(dep2.dep,tree)
                            dependency=Dep(dep2.dep,subj.dep,dtype,flag=u"XS")
                            #print dependency
                            new.append(dependency)
                    ## normal xsubj
                    dtype=self.decide_type(dep.dep,tree)
                    dependency=Dep(dep.dep,subj.dep,dtype,flag=u"XS")
                    #print dependency
                    new.append(dependency)
        for dep in new: # can't add these while iterating the same list, so these need to be added here
            tree.add_dep(dep)


#def merge_deps(sent,extra,conll_format=u"conll09"):
#    """ 
#    Sent is a list of lists (conll lines),
#    extra is a dictionary {key:dependent, value:list of (gov,dtype)} of new dependencies
#    """
#    # TODO: fill both columns ( HEAD and PHEAD, ...? )
#    form=formats[conll_format]
#    for key,value in extra.iteritems():
#        sent[key-1][form.HEAD]+=u","+u",".join(str(c[0]) for c in value)
#        sent[key-1][form.DEPREL]+=u","+u",".join(str(c[1]) for c in value)
#    return sent


#if __name__==u"__main__":

#    from optparse import OptionParser
#    parser = OptionParser()
#    parser.add_option("-f","--file",dest="f",default=None,help="Input file.")
#    parser.add_option("-o","--out",dest="o",default=None,help="Output file.")
#    (options, args) = parser.parse_args()
#    
#    if options.f is None:
#        print >> sys.stderr, "No input file."
#        sys.exit(1)

#    if options.o is None:
#        fName,end=options.f.rsplit(u".",1)
#        out=fName+u"_w_sec."+end
#    else: out=options.o

#    print >> sys.stderr, "Output file:",out
#        
#    
#    relmodel=Model(u"models/model_rel",u"models/feature_dict_rel.pkl",RELCLASSES)
#    relfunc=Relativizers(relmodel)
#    #jumpmodel=Model(u"models/model_jump",u"models/feature_dict_jump.pkl",JUMPCLASSES)
#    ccprop=ConjPropagation()
#    
#    xsubj=Xsubjects()

#    reader=FileReader()
#    writer=FileWriter(out)
#    tsent=0

#    f=codecs.open(u"rel_train.txt",u"wt",u"utf-8")
#    f2=codecs.open(u"cc_train.txt",u"wt",u"utf-8")

#    for sent in reader.conllReader(options.f):
#        if len(sent)>1:

#            tree=Tree(sent)

#            relfunc.learn(tree,f)

#            ccprop.learn(tree,f2)            

#            #relfunc.predict(tree) # rel

#            #xsubj.predict(tree) # xsubj

#            #ccprop.predict(tree) # jump
#            
#            #xsubj.predict(tree) # xsubj

#            #tree.tree_to_conll()

#        #writer.write_sent(sent)
#        tsent+=1
#        #if tsent>50000: break
#    print >> sys.stderr, "Processed sentences:",tsent

#    f.close()
#    f2.close()
        
                

