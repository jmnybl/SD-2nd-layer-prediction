#-*- coding: utf-8 -*-

# Order:
# 1) rel
# 2) xsubj
# 3) jump
# 4) xsubj


import codecs
import glob
import sys
import os
import cPickle
import math
import subprocess # needed for pigz
from collections import defaultdict

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
        self.readModel(modelf,featf)

    def readModel(self,modelFile,fFile):
        print >> sys.stdout, "Reading SVM model..."
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
        print >> sys.stdout,"Model:",modelFile
        f=codecs.open(fFile,"rb")
        self.fDict=cPickle.load(f)
        f.close()
        print >> sys.stdout,"Feature dictionary:",fFile
#        f=codecs.open(klassFile,"rb")
#        klasses=cPickle.load(f)
#        f.close()
        ## convert from key:ArgName,value:klassNum to key:klassNum,value:ArgName
        self.number2klass={}
        for key,value in self.klasses.iteritems():
            self.number2klass[value]=key



types=set([u"cc",u"conj",u"punct",u"ellipsis"])


def create_tree(sent):
    tree=[]
    for tok in sent:
        if tok[8]==u"0": continue # skip root
        govs=tok[8].split(u",")
        dtypes=tok[10].split(u",")
        dep=int(tok[0])
        for gov,dtype in zip(govs,dtypes):
            tree.append((int(gov),dep,dtype))
    return tree

### build a set of dep(conj) gov(conj)
def can_jump(dep,tree):
    if dep[2] in types: return False
    for g,d,t in tree:
        if g==dep[1] and t==u"conj":
            return True
        if g==dep[0] and t==u"conj":
            return True
    return False

#Takes one dependency(governor,dependent,type) and tree
#Returns a list of places(g,d), where possible new dependency can be
def possible_jumps(gov,dep,tree):
    candidates=set()
    #checks if there are possible_jumping-place with same governor, but different dependent
    for g,d,t in tree:
        if g==dep and t==u"conj":
            newDeps_g=gov
            newDeps_d=d
            candidates.add((newDeps_g,newDeps_d))    
    #checks if there are possible_jumping-place with same dependent, but different governor
    for g,d,t in tree:
        if g==gov and t==u"conj":
            newDeps_g=d
            newDeps_d=dep
            candidates.add((newDeps_g,newDeps_d))        
    return candidates


#Takes one possible new dependent place and checks if there are such a dep in tree
#Returns type of dep if found, otherwise None
def is_dep((possible_g,possible_d),tree):
    for g,d,t in tree:
        if g==possible_g and d==possible_d:
            return t
    return None


# Recursive search for all possible rec jumps
def gather_all_jumps(g,d,tree):
    recs=possible_jumps(g,d,tree)
    if len(recs)>0:
        new_set=set()
        for jump in recs:
            new_set|=gather_all_jumps(jump[0],jump[1],tree)
        recs|=new_set
        return recs
    else:
        return recs


def predict_one(model,features):
    maxscore=None
    klassnum=None
    for i in xrange(1,len(model.klasses)):
        score=0.0
        offset=((i-1)*model.base_features)
        for fnum,value in features: # list of feature numbers
            w=1/(math.sqrt(len(features)))
            pos=offset+fnum-1 # position in model.vector
            score+=w*model.vector[pos]
            if (maxscore is None) or maxscore<score:
                maxscore=score
                klassnum=i
    return klassnum

def jump_sentence(model,sent,fClass):
    """ Jump one conll sentence. """
    tree=create_tree(sent)
    jumped=defaultdict(lambda:[])
    for g,d,t in tree:
        if can_jump((g,d,t),tree):
            new_deps=gather_all_jumps(g,d,tree)
            uniq_set=set(new_deps)
            for jump_dep in uniq_set:
                features=fClass.createFeatures((g,d,t),jump_dep,tree,sent) #...should return (name,value) tuples
                fnums=[]
                for feature in features:
                    feat=model.fDict.get(feature)
                    if feat is not None:
                        fnums.append((feat,1.0))
                klass=predict_one(model,fnums)      
                if klass==1: continue
                klass_str=model.number2klass[klass]+u"JUMPED" #...for debugging
                jumped[jump_dep[1]].append((jump_dep[0],klass_str))
    print len(jumped)
    return jumped

def add_rels(model,sent,fClass):
    tree=create_tree(sent)
    new_deps=defaultdict(lambda:[])
    for g,d,t in tree:
        if t==u"rel":
            features=fClass.createFeatures((g,d),tree,sent)
            fnums=[]
            for feature in features:
                feat=model.fDict.get(feature)
                if feat is not None:
                    fnums.append((feat,1.0))
            klass=predict_one(model,fnums)
            klass_str=model.number2klass[klass]+u"extrarel" #...for debugging
            new_deps[d].append((g,klass_str))
    print u"new rels:",len(new_deps)
    return new_deps


def decide_type(token,tree,sent):
    """ token: gov of new dependency (and dep of xcomp also) """
    for g,d,t in tree:
        if g==token and t==u"cop":
            return u"xsubj-cop"
    pos=sent[token-1][4]
    if pos==u"Adv" or pos==u"N" or pos==u"A":
        return u"xsubj-cop"
    return u"xsubj"


def predict_xsubjects(sent):

    tree=create_tree(sent)
    new_deps=defaultdict(lambda:[])
    subjs={}
    for g,d,t in tree:
        if (t==u"nsubj" or t==u"nsubj-cop"):
            subjs.setdefault(g,[]).append((g,d,t))
    for g,d,t in tree:
        if t==u"xcomp" and g in subjs:
            for subj in subjs[g]:
                if (is_dep((d,subj[1]),tree)==u"xsubj" or is_dep((d,subj[1]),tree)==u"xsubj-cop"): break # this is because we run xsubj part twice!
                for go,de,ty in tree:
                    if go==d and ty==u"xcomp": 
                        if (is_dep((de,subj[1]),tree)==u"xsubj" or is_dep((de,subj[1]),tree)==u"xsubj-cop"): break
                        ## xcomp chain
                        dtype=decide_type(de,tree,sent)
                        new_deps[subj[1]].append((de,dtype))
                ## normal xsubj
                dtype=decide_type(d,tree,sent)
                new_deps[subj[1]].append((d,dtype))
    print u"new xsubjs:",len(new_deps)
    return new_deps


class FileReader(object):

    def _init__(self):
        pass
    

    def conllReader(self,fName):
        """ Returns one conll format sentence at a time. """
        data=self.readGzip(fName)
        sentence=[]
        for line in data:
            line=unicode(line,u"utf-8")
            line=line.strip()
            if not line or line.startswith(u"#"): # skip
                continue
            if line.startswith(u"1\t") and sentence:
                yield sentence
                sentence=[]
            sentence.append(line.split(u"\t"))
        else:
            if sentence:
                yield sentence

    def readGzip(self,fName):
        """ Uses multithreaded gzip implementation (pigz) to read gzipped files. """
        p=subprocess.Popen(("pigz","--decompress","--to-stdout","--processes","2",fName),stdout=subprocess.PIPE,stdin=None,stderr=subprocess.PIPE)
        return p.stdout

class FileWriter(object):
    """ Write the output file on-the-fly and keep track of progress. """

    def __init__(self,fName):
        self.f=codecs.open(fName,u"wt",u"utf-8")

    def write_sent(self,sent):
        """ Sent is a list of lists. """
        for line in sent:
            string=u"\t".join(c for c in line)
            self.f.write(string+u"\n")
        self.f.write(u"\n")
        


def merge_deps(sent,extra):
    """ 
    Sent is a list of lists (conll lines),
    extra is a dictionary {key:dependent, value:list of (gov,dtype)} of new dependencies
    """
    for key,value in extra.iteritems():
        sent[key-1][8]+=u","+u",".join(str(c[0]) for c in value)
        sent[key-1][10]+=u","+u",".join(str(c[1]) for c in value)
    return sent


if __name__==u"__main__":

    relmodel=Model(u"models/model_rel",u"models/feature_dict_rel.pkl",RELCLASSES)
    rel_feat=RelFeatures()

    #jumpmodel=Model(u"models/model_jump",u"models/feature_dict_jump.pkl",JUMPCLASSES)
    #jump_feat=JumpFeatures()
    #jumpmodel=None

    reader=FileReader()
    writer=FileWriter(u"jumped.conll")
    for sent in reader.conllReader(u"/home/jmnybl/ParseBank/parsebank_v3.conll09.gz"):
        if len(sent)>1:
            rels=add_rels(relmodel,sent,rel_feat)
            sent=merge_deps(sent,rels)
            xsubjs=predict_xsubjects(sent)
            sent=merge_deps(sent,xsubjs)
            #jumped=jump_sentence(jumpmodel,sent,jump_feat)
            #sent=merge_deps(sent,jumped)
        writer.write_sent(sent)
        
                

