# -*- coding: utf-8 -*-
import codecs
import math
import json
import collections
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
        features=set()
        
        ## DEPENDENCY RELATED (prefix:dep) ##

        #dependencyType
        depType=u"dep:dependencyType-"+dependency.dtype
        features.add(depType)

        #different governor
        if dependency.gov!=g: features.add(u"dep:hasDifferentGov")
        else: features.add(u"dep:hasSameGov")

        #if there already is a dependency of same type
        #two cases: 1) kissa ja koira syö --> verb can have two nsubj 2) kissa syö ja koira syö --> verb can't have two nsubj
        #this is only for case 2 
        for dep in tree.childs[g]:
            if dep.gov!=dependency.gov and dep.flag!=u"CC" and dep.dtype==dependency.dtype:
                features.add(u"dep:isSameType")
                break

        # is candidate going to same direction as original
        orig=dependency.gov.index-dependency.dep.index
        candidate=g.index-d.index
        if orig > 0 and candidate > 0: features.add(u"dep:bothGoingToLeft")
        elif orig < 0 and candidate < 0: features.add(u"dep:bothGoingToRight")
        else: features.add(u"dep:differentDirection")
            

        conjs=0
        #All dependencies Candidate deptok/govtok governs, and Dependency governing Original governor
        for dep in tree.childs[g]:
            if dep.flag!=u"CC": 
                features.add(u"dep:CandGovernorsDependency-"+dep.dtype)
            
        for dep in tree.childs[d]:
            if dep.flag!=u"CC":
                features.add(u"dep:CandDependentsDependency-"+dep.dtype)
        for dep in tree.childs[dependency.gov]:
            if dep.flag!=u"CC" and dep.dep!=dependency.dep:
                features.add(u"dep:OrigGovernorsDep-"+dep.dtype)
                if dep.dtype==u"cc":
                    features.add(u"dep:CCtoken-"+dep.dep.lemma)
            if dep.dtype==u"conj":
                conjs+=1
        for dep in tree.childs[dependency.dep]:
            if dep.dtype==u"cc":
                features.add(u"dep:CCtoken-"+dep.dep.lemma)
            elif dep.dtype==u"conj":
                conjs+=1
        

        assert conjs>0
        features.add(u"calc:conjs-"+unicode(conjs))
        
        
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
    moreFeatures=set()
    features=sorted(features)
    f_index=0
    #xrange
    while f_index<len(features):
        for i in xrange(f_index,len(features)):
            new_feature=features[f_index]+features[i]
            moreFeatures.add(new_feature)
        f_index+=1
    return moreFeatures

  
#create lemma and morpho features for given token
cats=set([u"CASE",u"NUM",u"SUBCAT",u"PRS",u"VOICE",u"INF"])
def give_morpho(token,f_list,prefix):
    """
    token: Token()
    """

    #FG: make these into sets and move it out of the function
    #cats=[u"CASE",u"NUM",u"SUBCAT",u"PRS",u"VOICE",u"INF"]

    #POSlist=[u"N",u"V",u"Adv",u"A",u"Num",u"Pron",u"Interj",u"C",u"Adp",u"Pcle",u"Punct",u"Null",u"Trash",u"Symb",u"Foreign"]
    #FeatList=[u"Nom",u"Gen",u"Par",u"Tra",u"Ess",u"Ine",u"Ela",u"Ill",u"Ade",u"Abl",u"All",u"Ins",u"Abe",u"Com"]


    f_list.add(prefix+u"Lemma-"+token.lemma)
    f_list.add(prefix+u"POS-"+token.pos)
    tags=token.feat.split(u"|")
    cat=None
    for tag in tags:
        if tag==u"_":continue
        if u"+" in tag: continue #just skip
        if u"_" in tag:
            cols=tag.split(u"_",1)
            cat,tag=cols[0],cols[1]
        if (cat is not None) and (cat in cats):
            f_list.add(prefix+u"Feat-"+tag)
            cat=None


#Takes a sourcefile(features_textfile) and converts that to numbers
#if loadDict == True, permission to load cPickled dict from disc and use it.
#if it's False, do not load dictionary but create a new and store it to disc
def convert_toNumbers(loadDict,task,dir):
    if loadDict:
        with open(os.path.join(dir,task,u"fnums.pkl"),"r") as f:
            featureDict=json.load(f)
        with open(os.path.join(dir,task,u"classes.pkl"),"r") as f:
            classDict=json.load(f)
    else:
        featureDict={}
        classDict={}
    sourcefile=codecs.open(os.path.join(dir,task,u"train.txt"),"rt","utf-8")
    targetfile=codecs.open(os.path.join(dir,task,u"train.num"),"wt","utf-8")
    regex=re.compile(u"\:[0-9]+$")
    for line in sourcefile:
        features=[]
        line=line.strip()
        columns=line.split()
        for i in range(1,len(columns)):
            fNumber=featureDict.setdefault(columns[i], len(featureDict)+1)
            features.append(fNumber)
        f_weight=1/(math.sqrt(len(features)))
        features.sort()
        features_str=u" ".join(unicode(feature)+u":"+unicode(f_weight) for feature in features)
        cNumber=classDict.setdefault(columns[0], len(classDict)+1)
        targetfile.write(unicode(cNumber)+u" "+features_str+"\n")
    sourcefile.close()
    targetfile.close()
    if loadDict==False:
        with open(os.path.join(dir,task,"fnums.pkl"),"w") as f:
            json.dump(featureDict,f)
        with open(os.path.join(dir,task,"classes.pkl"),"w") as f:
            json.dump(classDict,f)


