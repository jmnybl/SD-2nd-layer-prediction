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
from features import JumpFeatures,XsubjFeatures
import numpy as np

## ..not needed with sklearn... ##
#class Model(object):

#    def __init__(self,dir,task):
#        self.klasses=None ## readModel() fills these...
#        self.vector=None 
#        self.base_features=None
#        self.fDict=None
#        self.number2klass=None
#        self.readModel(dir,task)       

#    def readModel(self,dir,task):
#        print >> sys.stderr, u"Reading SVM model..."
#        if os.path.exists(os.path.join(dir,task,"vector.npy")) and os.path.exists(os.path.join(dir,task,"basef.json")):
#            self.vector=np.load(os.path.join(dir,task,"vector.npy"))
#            with open(os.path.join(dir,task,u"basef.json"),u"r") as f:
#                self.base_features=json.load(f)
#            print >> sys.stderr,u"Model:",os.path.join(dir,task,u"vector.npy")
#        else:
#            f=codecs.open(os.path.join(dir,task,u"model.svm"),u"rt",u"utf-8")
#            self.vector=None
#            for line in f:
#                line=line.strip()
#                if u"number of base features" in line:
#                    baseFs=int(line.split()[0])
#                    self.base_features=baseFs
#                elif u"highest feature index" in line:
#                    self.vector=np.zeros(int(line.split()[0]))
#                elif u"qid:0" in line:
#                    cols=line.split()
#                    del cols[0:2]
#                    del cols[-1]
#                    for col in cols:
#                        parts=col.split(u":")
#                        fNum=int(parts[0])
#                        value=float(parts[1])
#                        self.vector[fNum]=value
#            f.close()
#            np.save(os.path.join(dir,task,"vector.npy"),np.frombuffer(self.vector)) # save vector so we don't have to compile it again
#            with open(os.path.join(dir,task,u"basef.json"),u"w") as f:
#                json.dump(self.base_features,f)
#            print >> sys.stderr,u"Model:",os.path.join(dir,task,u"model.svm")
#        with open(os.path.join(dir,task,u"fnums.json"),u"r") as f:
#            self.fDict=json.load(f)
#        with open(os.path.join(dir,task,u"classes.json"),u"r") as f:
#            self.klasses=json.load(f)
#        ## convert from key:ArgName,value:klassNum to key:klassNum,value:ArgName
#        self.number2klass={}
#        for key,value in self.klasses.iteritems():
#            self.number2klass[value]=key

#    def predict_one(self,features):
#        """ features are in text format, first convert to numbers """
#        fnums=[]
#        for feature in features:
#            feat=self.fDict.get(feature)
#            if feat is not None:
#                fnums.append(feat)
#        
#        w=1/(math.sqrt(len(fnums)))
#        
#        maxscore=None
#        klassnum=None
#        for i in xrange(1,len(self.klasses)):
#            score=0.0
#            offset=((i-1)*self.base_features)
#            for fnum in fnums: # list of feature numbers
#                dim=offset+fnum # position in model.vector
#                score+=w*self.vector[dim]
#            if (maxscore is None) or maxscore<score:
#                maxscore=score
#                klassnum=i
#        return klassnum



def is_dep(g,d,tree):
    """ Check if there is dependency between g and d, and return list of types if found. """
    types=[]
    for dep in tree.deps:
        if g==dep.gov and d==dep.dep and dep.dtype!=u"name": # TODO: name?
            types.append(dep.dtype)
    return types

#Writes the textfile, which has to be later converted to numbers and feed to svm.
#def writeData(f,klass,features):
#    feat_str=u" ".join(feat for feat in features)
#    f.write(klass+u" "+feat_str+u"\n")


class ConjPropagation(object):

    def __init__(self,model=None,vectorizer=None):
        self.features=JumpFeatures()
        self.model=model
        self.vectorizer=vectorizer
        self.resttypes=set([u"cc",u"conj",u"punct",u"ellipsis","orphan"])
        self.allowed_types=[u"aux:pass", u"acl:relcl"]

    def can_jump(self,dep,tree):
        """ Check whether dep can jump. """
        if dep.dtype in self.resttypes: return False
        for conj in tree.conjs:
            if dep.dep==conj.gov or dep.gov==conj.gov: # is connected to 'conj'
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


    def gather_all_jumps(self,g,d,tree):
        """ Recursive search for all possible jumps. (g,d) is one candidate for propagation, return a set of all possible new dependencies this candidate can produce. """
        recs=self.possible_jumps(g,d,tree)
        if len(recs)>0:
            new_set=set()
            for gov,dep in recs:
                new_set|=self.gather_all_jumps(gov,dep,tree)
            recs|=new_set
            return recs
        else:
            return recs


    def learn(self,tree):
        #candidates=[dep for key,dep in tree.basedeps.iteritems()]
        examples=[]
        labels=[]
        candidates=[dep for dep in tree.deps if dep.flag!=u"CC"] # do not propagate propagated, always use the original
        for dep in candidates:
            if self.can_jump(dep,tree):
                new_deps=self.gather_all_jumps(dep.gov,dep.dep,tree)
                for g,d in new_deps:
                    types=is_dep(g,d,tree)
                                       
                    if not types:
                        klass=u"no"
                    else:
                        if len(types)>1 or types[0] in self.resttypes:
                            print >> sys.stderr, "more than one possible relation type or restricted relation type, skipping", types, tree
                            continue
                        klass=types[0]
                        if ":" in klass and klass not in self.allowed_types:
                            klass=klass.split(u":",1)[0]
                    features=self.features.create(dep,g,d,tree,self.allowed_types)
                    examples.append(features)
                    labels.append(klass)
        return examples, labels
                    

    def predict(self,tree):
        """ Jump one tree. """
        if self.model is None:
            print >> sys.stderr, u"no model found"
            sys.exit(1)
        new=[]
        # candidates=get_candidates(tree)
        candidates=[dep for dep in tree.deps if dep.flag!=u"CC"]
        for dep in candidates: #tree.deps
            if self.can_jump(dep,tree):
                new_deps=self.gather_all_jumps(dep.gov,dep.dep,tree)
                for g,d in new_deps:
                    features=self.features.create(dep,g,d,tree,self.allowed_types)
                    klass=self.model.predict(self.vectorizer.transform(features))[0]
                    if klass==u"no": continue
                    dependency=Dep(g,d,klass,flag=u"CC")
                    new.append(dependency)
        for dep in new:
            tree.add_dep(dep)



#class Relativizers(object):

#    def __init__(self,model=None):
#        self.features=RelFeatures()
#        self.model=model

#    def learn(self,tree,outfile):
#        for rel in tree.rels:
#            features=self.features.create(rel.gov,rel.dep,tree)
#            types=is_dep(rel.gov,rel.dep,tree)
#            if len(types)<2: continue # post-processed dependency, do not use in training
#            for dtype in types:
#                if dtype==u"rel": continue
#                klass=dtype.split(u"&",1)[1]
#                writeData(outfile,klass,features)
#                break


#    def post_process_rel(self,g,d,t,tree):
#        # move dependency if iccomp and it does not have this dtype already
#        for dep in tree.deps:
#            if dep.gov==g and dep.dtype==u"iccomp":
#                deps=set()
#                for dep2 in tree.childs[dep.dep]: # dependents of iccomp
#                    deps.add(dep2.dtype) # collect all dependents of iccomp
#                if t not in deps:
#                    g=dep.dep
#                    break
#        return g,d,t

#    def predict(self,tree):
#        if self.model is None:
#            print >> sys.stderr, u"no model found"
#            sys.exit(1)
#        for rel in tree.rels:
#            features=self.features.create(rel.gov,rel.dep,tree)
#            klass=self.model.predict_one(features)
#            klass_str=u"rel&"+self.model.number2klass[klass]
#            g,d,t=self.post_process_rel(rel.gov,rel.dep,klass_str,tree)
#            dependency=Dep(g,d,t,flag=u"REL")
#            #print dependency
#            tree.add_dep(dependency)


class Xsubjects(object):

    def __init__(self,model=None,vectorizer=None):
        self.features=XsubjFeatures()
        self.model=model
        self.vectorizer=vectorizer
        self.allowed_types=[u"aux:pass", u"acl:relcl"] # allowed language specific dep types, all rest are cut to universal

    def decide_type(self,token,tree):
        """ token: gov of new dependency (and dep of xcomp also) """
        # if this is not verb nor copula --> no xsubj
        for dep in tree.childs[token]:
            if dep.dtype==u"cop":
                return u"nsubj:cop"
        if token.cpos!=u"VERB" and token.cpos!=u"AUX":
            return None
        if u"VerbForm=Part" in token.feat and (u"Case=Ess" in token.feat or u"Case=Tra" in token.feat): # ...puut ovat uhattuina, alue tulee kartoitetuksi (acomp like structures)
            return None
        return u"nsubj"


    def learn(self,tree):
        examples=[]
        labels=[]
        for dep in tree.deps:
            if (dep.dtype==u"xcomp" or dep.dtype==u"xcomp:ds") and dep.gov in tree.subjs:
                for subj in tree.subjs[dep.gov]:
                    types=is_dep(dep.dep,subj.dep,tree)
                    if (u"nsubj" in types) or (u"nsubj:cop" in types):
                        label=1
                    else:
                        label=0
                    # TODO collect features
                    features=self.features.create(dep.gov,dep.dep,subj.dep,tree,self.allowed_types) # xcomp_g,new_g,new_d,tree
                    examples.append(features)
                    labels.append(label)
                    for dep2 in tree.childs[dep.dep]: # xcomp chain
                        if dep2.dtype==u"xcomp" or dep2.dtype==u"xcomp:ds":
                            types=is_dep(dep2.gov,subj.dep,tree)
                            if (u"nsubj" in types) or (u"nsubj:cop" in types): # positive!
                                label=1
                            else:
                                label=0
                            # TODO collect features
                            features=self.features.create(dep2.gov,dep2.dep,subj.dep,tree,self.allowed_types) # xcomp_g,new_g,new_d,tree
                            examples.append(features)
                            labels.append(label)
                            #dtype=self.decide_type(dep2.dep,tree)
                            #if not dtype:
                            #    continue
                            #dependency=Dep(dep2.dep,subj.dep,dtype,flag=u"XS")
                            #print dependency
                            #new.append(dependency)
                    ## normal xsubj
                    #dtype=self.decide_type(dep.dep,tree)
                    #if not dtype:
                    #    continue
                    #dependency=Dep(dep.dep,subj.dep,dtype,flag=u"XS")
                    #print dependency
                    #new.append(dependency)
        #for dep in new: # can't add these while iterating the same list, so these need to be added here
        #    tree.add_dep(dep)
        return examples, labels


    def predict(self,tree):
        if self.model is None:
            print >> sys.stderr, u"no model found"
            sys.exit(1)
        new=[]
        for dep in tree.deps:
            if (dep.dtype==u"xcomp" or dep.dtype==u"xcomp:ds") and dep.gov in tree.subjs:
                for subj in tree.subjs[dep.gov]:
                    types=is_dep(dep.dep,subj.dep,tree)
                    if u"nsubj:xsubj" in types: continue # this is because we run xsubj part twice!
                    features=self.features.create(dep.gov,dep.dep,subj.dep,tree,self.allowed_types) # xcomp_g,new_g,new_d,tree
                    label=self.model.predict(self.vectorizer.transform(features))[0]
                    if label==1:
                        dependency=Dep(dep.dep,subj.dep,"nsubj:xsubj",flag=u"XS")
                        new.append(dependency)

                        for dep2 in tree.childs[dep.dep]: # xcomp chain
                            if (dep2.dtype==u"xcomp" or dep2.dtype==u"xcomp:ds"):
                                types=is_dep(dep2.gov,subj.dep,tree)
                                if u"nsubj:xsubj" in types: continue
                                features=self.features.create(dep2.gov,dep2.dep,subj.dep,tree,self.allowed_types) # xcomp_g,new_g,new_d,tree
                                label=self.model.predict(self.vectorizer.transform(features))[0]
                                if label==1:
                                    dependency=Dep(dep2.dep,subj.dep,"nsubj:xsubj",flag=u"XS")
                                    new.append(dependency)
                                
                    
        for dep in new: # can't add these while iterating the same list, so these need to be added here
            tree.add_dep(dep)

                

