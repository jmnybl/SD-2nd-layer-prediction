#-*- coding: utf-8 -*-

# Order:
# 1) rel
# 2) xsubj
# 3) jump
# 4) xsubj

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

CoNLLFormat=namedtuple("CoNLLFormat",["ID","FORM","LEMMA","POS","FEAT","HEAD","DEPREL"])

#Column lists for the various formats
formats={"conll09":CoNLLFormat(0,1,2,4,6,8,10)}


def create_tree(sent,conll_format=u"conll09"):
    form=formats[conll_format]
    tree=[]
    for tok in sent:
        if tok[form.HEAD]==u"0": continue # skip root
        govs=tok[form.HEAD].split(u",")
        dtypes=tok[form.DEPREL].split(u",")
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
    for g,d,t in tree:
        if g==dep and t==u"conj": # dependent can jump
            newDeps_g=gov
            newDeps_d=d
            candidates.add((newDeps_g,newDeps_d))    
        elif g==gov and t==u"conj": # governor can jump
            newDeps_g=d
            newDeps_d=dep
            candidates.add((newDeps_g,newDeps_d))        
    return candidates


def is_dep((possible_g,possible_d),tree):
    """ Check if there is dependency between g and d, and return type if found. """
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
            pos=offset+fnum # position in model.vector
            score+=w*model.vector[pos]
            if (maxscore is None) or maxscore<score:
                maxscore=score
                klassnum=i
    return klassnum

def merge_rels(tree):
    merged=[]
    rels=[]
    for g,d,t in tree:
        if t==u"rel":
            rels.append((g,d,t))
    must_delete=[]
    for g,d,t in tree:
        if (g,d,u"rel") in rels and t!=u"rel": # merge only if both g and d same, not if post_processed
            dtype=u"rel&"+t
            merged.append((g,d,dtype))
            must_delete.append((g,d,u"rel"))
        else: merged.append((g,d,t)) # add as is
    for depen in must_delete: # because we also added original rel dependencies, we now have to remove those
        merged.remove(depen)
    return merged

def jump_sentence(model,sent,fClass):
    """ Jump one conll sentence. """
    tree=create_tree(sent)
    treec=merge_rels(tree)
    jumped=defaultdict(lambda:[])
    for g,d,t in treec:
        if can_jump((g,d,t),treec):
            new_deps=gather_all_jumps(g,d,treec)
            uniq_set=set(new_deps)
            for jump_dep in uniq_set:
                features=fClass.createFeatures((g,d,t),jump_dep,treec,sent) #...should return (name,value) tuples
                fnums=[]
                for feature in features:
                    feat=model.fDict.get(feature)
                    if feat is not None:
                        fnums.append((feat,1.0))
                klass=predict_one(model,fnums)      
                if klass==1: continue
                klass_str=model.number2klass[klass]
                if u"&" in klass_str: # this is merged rel, split
                    rel,sec=klass_str.split(u"&",1)   
                    jumped[jump_dep[1]].append((jump_dep[0],rel))
                    jumped[jump_dep[1]].append((jump_dep[0],sec))
                else: jumped[jump_dep[1]].append((jump_dep[0],klass_str))
    return jumped

def post_process_rel(g,d,t,tree):
    # move dependency if iccomp and it does not have this dtype already
    for gov,dep,type in tree:
        deps=set()
        if g==gov and type==u"iccomp":
            for gov2,dep2,type2 in tree:
                if gov2==dep:
                    deps.add(type2) # collect all dependents of iccomp
            if t not in deps:
                g=dep 
                break
    return g,d,t

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
            klass_str=model.number2klass[klass]
            gov,dep,ty=post_process_rel(g,d,klass_str,tree)
            new_deps[dep].append((gov,ty))
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
                for go,de,ty in tree: # xcomp chain
                    if go==d and ty==u"xcomp": 
                        if (is_dep((de,subj[1]),tree)==u"xsubj" or is_dep((de,subj[1]),tree)==u"xsubj-cop"): break
                        dtype=decide_type(de,tree,sent)
                        new_deps[subj[1]].append((de,dtype))
                ## normal xsubj
                dtype=decide_type(d,tree,sent)
                new_deps[subj[1]].append((d,dtype))
    return new_deps


class FileReader(object):

    def _init__(self):
        pass
    

    def conllReader(self,fName):
        """ Returns one conll format sentence at a time. """
        if fName.endswith(u".gz"):
            data=self.readGzip(fName)
        else:
            data=codecs.open(fName,u"rt",u"utf-8")
        sentence=[]
        for line in data:
            if not isinstance(line,unicode):
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
        self.end=fName.rsplit(u".",1)[1]
        if fName.endswith(u".gz"):
            self.f=gzip.open(fName,u"wb")
        else: self.f=codecs.open(fName,u"wt",u"utf-8")

    def write_sent(self,sent):
        """ Sent is a list of lists. """
        for line in sent:
            string=u"\t".join(c for c in line)+u"\n"
            if self.end==u"gz":
                string=string.encode(u"utf-8")
            self.f.write(string)
        if self.end==u"gz":
            self.f.write((u"\n").encode(u"utf-8"))
        else: self.f.write(u"\n")
        


def merge_deps(sent,extra,conll_format=u"conll09"):
    """ 
    Sent is a list of lists (conll lines),
    extra is a dictionary {key:dependent, value:list of (gov,dtype)} of new dependencies
    """
    # TODO: fill both columns ( HEAD and PHEAD, ...? )
    form=formats[conll_format]
    for key,value in extra.iteritems():
        sent[key-1][form.HEAD]+=u","+u",".join(str(c[0]) for c in value)
        sent[key-1][form.DEPREL]+=u","+u",".join(str(c[1]) for c in value)
    return sent


if __name__==u"__main__":

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f","--file",dest="f",default=None,help="Input file.")
    parser.add_option("-o","--out",dest="o",default=None,help="Output file.")
    (options, args) = parser.parse_args()
    
    if options.f is None:
        print >> sys.stderr, "No input file."
        sys.exit(1)

    if options.o is None:
        fName,end=options.f.rsplit(u".",1)
        out=fName+u"_w_sec."+end
    else: out=options.o

    print >> sys.stderr, "Output file:",out
        
    
    start=time.clock()

    relmodel=Model(u"models/model_rel",u"models/feature_dict_rel.pkl",RELCLASSES)
    rel_feat=RelFeatures()

    jumpmodel=Model(u"models/model_jump",u"models/feature_dict_jump.pkl",JUMPCLASSES)
    jump_feat=JumpFeatures()
    #jumpmodel=None

    end=time.clock()

    print >> sys.stderr, "%.2gm" % ((end-start)/60)

    reader=FileReader()
    writer=FileWriter(out)
    tsent=0
    for sent in reader.conllReader(options.f):
        if len(sent)>1:
            rels=add_rels(relmodel,sent,rel_feat) # rel
            sent=merge_deps(sent,rels)
            xsubjs=predict_xsubjects(sent) # xsubj
            sent=merge_deps(sent,xsubjs)
            jumped=jump_sentence(jumpmodel,sent,jump_feat) # jump
            sent=merge_deps(sent,jumped)
            xsubjs=predict_xsubjects(sent) # xsubj
            sent=merge_deps(sent,xsubjs)
        writer.write_sent(sent)
        tsent+=1
        #if tsent>50000: break
    print "Processed sentences:",tsent
        
                

