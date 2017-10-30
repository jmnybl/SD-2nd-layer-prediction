# -*- coding: utf-8 -*-
import codecs
import math
import json
import re
import sys
import os

SCRIPTDIR=os.path.dirname(os.path.abspath(__file__))


class JumpFeatures:
#Class to handle all JumpFeature-based stuff.

    def __init__(self):
        pass


    def create(self,dependency,g,d,tree):
        """
        dependency: Dep(), original dep
        g: Token(), gov of propagated dependency
        d: Token(), dep of propagated dependency
        tree: Tree()
        """
        features={}
        
        ## DEPENDENCY RELATED (prefix:dep) ##

        #dependencyType
        depType=u"dep:dependencyType-"+dependency.dtype
        features[depType]=1

        #different governor
        if dependency.gov!=g: features[u"dep:hasDifferentGov"]=1
        else: features[u"dep:hasSameGov"]=1

        #if there already is a dependency of same type
        #two cases: 1) kissa ja koira syö --> verb can have two nsubj 2) kissa syö ja koira syö --> verb can't have two nsubj
        #this is only for case 2 
        for dep in tree.childs[g]:
            if dep.gov!=dependency.gov and dep.flag!=u"CC" and dep.dtype==dependency.dtype:
                features[u"dep:isSameType"]=1
                break

        # is candidate going to same direction as original
        orig=int(dependency.gov.index.split(".",1)[0])-int(dependency.dep.index.split(".",1)[0])
        candidate=int(g.index.split(".",1)[0])-int(d.index.split(".",1)[0])
        if orig > 0 and candidate > 0: features[u"dep:bothGoingToLeft"]=1
        elif orig < 0 and candidate < 0: features[u"dep:bothGoingToRight"]=1
        else: features[u"dep:differentDirection"]=1
            

        conjs=0
        #All dependencies Candidate deptok/govtok governs, and Dependency governing Original governor
        for dep in tree.childs[g]:
            if dep.flag!=u"CC": 
                features[u"dep:CandGovernorsDependency-"+dep.dtype]=1
            
        for dep in tree.childs[d]:
            if dep.flag!=u"CC":
                features[u"dep:CandDependentsDependency-"+dep.dtype]=1
        for dep in tree.childs[dependency.gov]:
            if dep.flag!=u"CC" and dep.dep!=dependency.dep:
                features[u"dep:OrigGovernorsDep-"+dep.dtype]=1
                if dep.dtype==u"cc":
                    features[u"dep:CCtoken-"+dep.dep.lemma]=1
            if dep.dtype==u"conj":
                conjs+=1
        for dep in tree.childs[dependency.dep]:
            if dep.dtype==u"cc":
                features[u"dep:CCtoken-"+dep.dep.lemma]=1
            elif dep.dtype==u"conj":
                conjs+=1
        

        assert conjs>0
        features[u"calc:conjs"]=conjs
        
        
        ## TOKEN RELATED (prefic:tok) ##
        
        
        # gov of candidate
        give_morpho(g,features,u"govtok:")
        # dep of candidate
        give_morpho(d,features,u"deptok:")

        # gov of original (if different than gov of candidate)
        if dependency.gov!=g:
            give_morpho(dependency.gov,features,u"Origgovtok:")
        # dep of original (if different than dep of candidate)
        if dependency.dep!=d:
            give_morpho(dependency.dep,features,u"Origdeptok:")
        
        
        ## CREATE ALL PAIRS ##
        features=createAllPairs(features)

        return features


    


class RelFeatures:

    def __init__(self):
        pass
    
    def create(self,g,d,tree):
        """
        g: Token()
        d: Token()
        tree: Tree()
        """
        features=set()
        features.add(u"DummyFeature")
        
        give_morpho(d,features,u"deptok:")
        give_morpho(g,features,u"govtok:")

        ## DEPENDENCIES (governed by deptok/govtok):
        for dep in tree.childs[g]:
            if dep.flag!=u"BASE" or dep.dtype==u"rel": continue
            features.add(u"dep:GovernorsDependency-"+dep.dtype)
        for dep in tree.childs[d]:
            if dep.flag!=u"BASE": continue
            features.add(u"dep:DependentsDependency-"+dep.dtype)
            
        features=createAllPairs(features)
        
        return features
    


####################  GENERAL FUNCTIONS   #######################
#################################################################

#Takes features and creates all possible pairs (f1f1, f1f2, f1f3, f2f2...)
def createAllPairs(features):
    feat_keys=sorted(features.keys())
    for f_index in xrange(len(feat_keys)):
        for i in xrange(f_index,len(feat_keys)):
            new_feature=feat_keys[f_index]+feat_keys[i]
            features[new_feature]=1
    return features

  
#create lemma and morpho features for given token
#cats=set([u"CASE",u"NUM",u"SUBCAT",u"PRS",u"VOICE",u"INF"])
def give_morpho(token,f_list,prefix):
    """
    token: Token()
    """

    f_list[prefix+u"Lemma-"+token.lemma]=1
    f_list[prefix+u"CPOS-"+token.cpos]=1
    if token.feat!=u"_":
        for tag in token.feat.split(u"|"):
            f_list[prefix+tag]=1
#    cat=None
#    for tag in tags:
#        if tag==u"_":continue
#        if u"+" in tag: continue #just skip
#        if u"_" in tag:
#            cols=tag.split(u"_",1)
#            cat,tag=cols[0],cols[1]
#        if (cat is not None) and (cat in cats):
#            f_list.add(prefix+u"Feat-"+tag)
#            cat=None


## ..not needed with sklearn.. ##
#Takes a sourcefile(features_textfile) and converts that to numbers
#if loadDict == True, permission to load cPickled dict from disc and use it.
#if it's False, do not load dictionary but create a new and store it to disc
#def convert_toNumbers(loadDict,task,dir):
#    if loadDict:
#        with open(os.path.join(dir,task,u"fnums.json"),u"r") as f:
#            featureDict=json.load(f)
#        with open(os.path.join(dir,task,u"classes.json"),u"r") as f:
#            classDict=json.load(f)
#    else:
#        featureDict={}
#        classDict={}
#    sourcefile=codecs.open(os.path.join(dir,task,u"train.txt"),"rt","utf-8")
#    targetfile=codecs.open(os.path.join(dir,task,u"train.num"),"wt","utf-8")
#    regex=re.compile(u"\:[0-9]+$")
#    for line in sourcefile:
#        features=[]
#        line=line.strip()
#        columns=line.split()
#        for i in range(1,len(columns)):
#            fNumber=featureDict.setdefault(columns[i], len(featureDict)+1)
#            features.append(fNumber)
#        f_weight=1/(math.sqrt(len(features)))
#        features.sort()
#        features_str=u" ".join(unicode(feature)+u":"+unicode(f_weight) for feature in features)
#        cNumber=classDict.setdefault(columns[0], len(classDict)+1)
#        targetfile.write(unicode(cNumber)+u" "+features_str+"\n")
#    sourcefile.close()
#    targetfile.close()
#    if loadDict==False:
#        with open(os.path.join(dir,task,"fnums.json"),u"w") as f:
#            json.dump(featureDict,f)
#        with open(os.path.join(dir,task,"classes.json"),u"w") as f:
#            json.dump(classDict,f)


