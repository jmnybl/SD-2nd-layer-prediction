#-*- coding: utf-8 -*-


import codecs
import glob
import sys
import os
import json
import math
import subprocess # needed for pigz
import gzip

from tree import Tree,Dep
from features import JumpFeatures,RelFeatures
import numpy as np


class Model(object):

    def __init__(self,dir,task):
        self.klasses=None ## readModel() fills these...
        self.vector=None 
        self.base_features=None
        self.fDict=None
        self.number2klass=None
        self.readModel(dir,task)       

    def readModel(self,dir,task):
        print >> sys.stderr, u"Reading SVM model..."
        if os.path.exists(os.path.join(dir,task,"vector.npy")) and os.path.exists(os.path.join(dir,task,"basef.json")):
            self.vector=np.load(os.path.join(dir,task,"vector.npy"))
            with open(os.path.join(dir,task,u"basef.json"),u"r") as f:
                self.base_features=json.load(f)
            print >> sys.stderr,u"Model:",os.path.join(dir,task,u"vector.npy")
        else:
            f=codecs.open(os.path.join(dir,task,u"model.svm"),u"rt",u"utf-8")
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
            np.save(os.path.join(dir,task,"vector.npy"),np.frombuffer(self.vector)) # save vector so we don't have to compile it again
            with open(os.path.join(dir,task,u"basef.json"),u"w") as f:
                json.dump(self.base_features,f)
            print >> sys.stderr,u"Model:",os.path.join(dir,task,u"model.svm")
        with open(os.path.join(dir,task,u"fnums.json"),u"r") as f:
            self.fDict=json.load(f)
        with open(os.path.join(dir,task,u"classes.json"),u"r") as f:
            self.klasses=json.load(f)
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
                fnums.append(feat)
        
        w=1/(math.sqrt(len(fnums)))
        
        maxscore=None
        klassnum=None
        for i in xrange(1,len(self.klasses)):
            score=0.0
            offset=((i-1)*self.base_features)
            for fnum in fnums: # list of feature numbers
                dim=offset+fnum # position in model.vector
                score+=w*self.vector[dim]
            if (maxscore is None) or maxscore<score:
                maxscore=score
                klassnum=i
        return klassnum



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


    def learn(self,tree,outfile):
        for dep in tree.deps:
            if dep.flag!=u"CC" and self.can_jump(dep,tree):
                if dep.dtype==u"rel":
                    continue
                new_deps=self.gather_all_jumps(dep.gov,dep.dep,tree)
                for g,d in new_deps:
                    types=is_dep(g,d,tree)                    
                    if not types:
                        klass=u"no"
                    elif len(types)==2 and u"rel" in types:
                        for t in types:
                            if t!=u"rel":
                                klass=t
                                break
                    else:
                        assert len(types)<2
                        klass=types[0]
                    features=self.features.create(dep,g,d,tree)
                    writeData(outfile,klass,features)
                    

    def predict(self,tree):
        """ Jump one tree. """
        if self.model is None:
            print >> sys.stderr, u"no model found"
            sys.exit(1)
        new=[]
        for dep in tree.deps:
            if self.can_jump(dep,tree):
                if dep.dtype==u"rel":
                    continue # this is just the rel, we want the secondary function
                new_deps=self.gather_all_jumps(dep.gov,dep.dep,tree)
                for g,d in new_deps:
                    features=self.features.create(dep,g,d,tree) #...should return (name,value) tuples
                    klass=self.model.predict_one(features)
                    klass_str=self.model.number2klass[klass]
                    if klass_str==u"no": continue
                    if u"&" in klass_str: # this is merged rel
                        #print >> sys.stderr, klass_str
                        dependency=Dep(g,d,u"rel",flag=u"CC") # add also rel
                        new.append(dependency)
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
            for dtype in types:
                if dtype==u"rel": continue
                klass=dtype.split(u"&",1)[1]
                writeData(outfile,klass,features)
                break


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
            klass_str=u"rel&"+self.model.number2klass[klass]
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

                

