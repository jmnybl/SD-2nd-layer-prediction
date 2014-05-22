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

from Features import JumpFeatures
import numpy as np


class Model(object):

    def __init__(self,modelf,featf):
        self.klasses={u"DontJump":1,u"adpos":2,u"advcl":3,u"advmod":4,u"acomp":5,u"amod":6,u"appos":7,u"aux":8,u"cc":9,u"ccomp":10,u"compar":11,
u"comparator":12,u"complm":13,u"conj":14,u"cop":15,u"csubj":16,u"csubj-cop":17,u"dep":18,u"det":19,u"dobj":20,u"ellipsis":21,u"gobj":22,u"iccomp":23,
u"infmod":24,u"mark":25,u"name":26,u"neg":27,u"nn":28,u"nommod":29,u"nsubj":30,u"nsubj-cop":31,u"num":32,u"number":33,u"parataxis":34,u"partmod":35,u"poss":36,u"preconj":37,u"prt":38,u"quantmod":39,u"rcmod":40,
u"rel":41,u"xcomp":42,u"xsubj":43,u"gsubj":44,u"nommod-own":45,u"intj":46,u"punct":47,u"auxpass":48,u"voc":49,u"xsubj-cop":50,u"rel&nsubj":51,u"rel&dobj":52,u"rel&nommod":53,u"rel&nsubj-cop":54,u"rel&advmod":55,u"rel&advcl":56,u"rel&nommod-own":57,u"rel&partmod":58,u"rel&poss":59,u"rel&xcomp":60} # TODO: create this during training and pickle
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



types=set((u"cc",u"conj",u"punct",u"ellipsis"))


def create_tree(sent):
    tree=[]
    for tok in sent:
        gov=int(tok[8])
        if gov==0: continue # skip root
        dep=int(tok[0])
        dtype=tok[10]
        tree.append((gov,dep,dtype))
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
    return_list=[]
    #checks if there are possible_jumping-place with same governor, but different dependent
    for g,d,t in tree:
        if g==dep and t==u"conj":
            newDeps_g=gov
            newDeps_d=d
            return_list.append((newDeps_g,newDeps_d))    
    #checks if there are possible_jumping-place with same dependent, but different governor
    for g,d,t in tree:
        if g==gov and t==u"conj":
            newDeps_g=d
            newDeps_d=dep
            return_list.append((newDeps_g,newDeps_d))        
    return return_list


#Takes one possible new dependent place and checks if there are such a dep in tree
#Returns type of dep if found, otherwise None
def is_jumped((possible_g,possible_d),tree):
    for g,d,t in tree:
        if g==possible_g and d==possible_d:
            return t
    return None


# Recursive search for all possible rec jumps
def gather_all_jumps(g,d,tree):
    recs=possible_jumps(g,d,tree)
    if len(recs)>0:
        list=[]
        for jump in recs:
            list+=gather_all_jumps(jump[0],jump[1],tree)
        recs=recs+list
        return recs
    else:
        return recs


def predict_jump(model,features):
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

def jump_sentence(model,sent):
    """ Jump one conll sentence. """
    fe=JumpFeatures()
    tree=create_tree(sent)
    jumped={}
    for g,d,t in tree:
        if can_jump((g,d,t),tree):
            new_deps=gather_all_jumps(g,d,tree) # TODO maybe should return a set
            uniq_set=set(new_deps)
            for jump_dep in uniq_set:
                features=fe.createFeatures((g,d,t),jump_dep,tree,sent) #...should return (name,value) tuples
                fnums=[]
                for feature in features:
                    feat=model.fDict.get(feature)
                    if feat is not None:
                        fnums.append((feat,1.0))
                    else: pass 
                    klass=predict_jump(model,fnums)
                        
                    if klass==1: continue
                    klass_str=model.number2klass[klass]
                    jumped[jump_dep[1]]=(jump_dep[0],klass_str)
    print len(jumped)
    return jumped



class FileWriter(object):
    """ Write the output file on-the-fly and keep track of progress. """
    def __init__(self):
        pass



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
            if not line:
                yield sentence
                sentence=[]
            else:
                sentence.append(line.split(u"\t"))
        data.close()
        if len(sentence)>0:
            yield sentence

    def readGzip(self,fName):
        """ Uses multithreaded gzip implementation (pigz) to read gzipped files. """
        p=subprocess.Popen(("pigz","--decompress","--to-stdout","--processes","2",fName),stdout=subprocess.PIPE,stdin=None,stderr=subprocess.PIPE)
        return p.stdout



if __name__==u"__main__":

    model=Model(u"models/model_jump",u"models/feature_dict_jump.pkl")

    reader=FileReader()
    for sent in reader.conllReader(u"/home/jmnybl/ParseBank/parsebank_v3.conll09.gz"):
        print sent
        jumped=jump_sentence(model,sent)
        for i in xrange(0,len(sent)):
            value=jumped.get(int(sent[i][0]))
            if value is not None:
                sent[i][8]+=u","+str(value[0])
                sent[i][10]+=u","+str(value[1])

        print sent
                

        


