# -*- coding: utf-8 -*-
import codecs
import math
import cPickle
import collections
import re
import sys
import os

#sys.path.append(os.path.expanduser("~/svn_checkout/annotator"))
sys.path.append("/home/jmnybl/svn_checkout/TDT/")
from dtreebank.core.omorfi_pos import tag_cat_dict


class JumpFeatures:
#Class to handle all JumpFeature-based stuff.

    def __init__(self):
        pass


    def createFeatures(self,dependency,place,tree,sent):
        """
        dependency is the original dep,
        place is the new (gov,dep) pair
        tree is a list of (gov,dep,dtype) triplets
        sent is a conll lines
        """
        features=[]
        
        ## DEPENDENCY RELATED (prefix:dep) ##

        #dependencyType
        depType=u"dep:dependencyType-"+dependency[2]
        features.append(depType)

        #different governor
        if dependency[0]!=place[0]: features.append(u"dep:hasDifferentGov")
        else: features.append(u"dep:hasSameGov")

        #if there already is a dependency of same type
        #two cases: 1) kissa ja koira syö --> verb can have two nsubj 2) kissa syö ja koira syö --> verb can't have two nsubj
        #this is only for case 2
        for g,d,t in tree: ###FG maybe get a list/dictionary to answer this question
            if g!=dependency[0]:
                if g==place[0] and t==dependency[2]:
                    features.append(u"dep:isSameType")
                    break

        # is candidate going to same direction as original
        orig=dependency[0]-dependency[1]
        candidate=place[0]-place[1]
        if orig > 0 and candidate > 0: features.append(u"dep:bothGoingToLeft")
        elif orig < 0 and candidate < 0: features.append(u"dep:bothGoingToRight")
        else: features.append(u"dep:differentDirection")
            

        conjs=0
        #All dependencies Candidate deptok/govtok governs, and Dependency governing Original governor
        for g,d,t in tree: ##FG - try to avoid these loops
            if g==place[1]:
                features.append(u"dep:CandDependentsDependency-"+t)
            if g==place[0]:
                features.append(u"dep:CandGovernorsDependency-"+t)
            if d==dependency[0]:
                features.append(u"dep:OrigGovernorsDep-"+t)

            #Lemma of cc
            #same governor -jump
            if g==dependency[1] and t==u"cc":
                lemma=sent[d-1][2]
                features.append(u"dep:CCtoken-"+lemma) 
            #different gov, same dependent -jump
            if g==dependency[0] and t==u"cc":
                lemma=sent[d-1][2]
                features.append(u"dep:CCtoken-"+lemma) 
        
            # conjs
            if g==dependency[0] and t==u"conj": conjs+=1
            elif g==dependency[1] and t==u"conj": conjs+=1
        features.append(u"calc:conjs-"+unicode(conjs))
        
        
        ## TOKEN RELATED (prefic:tok) ##
        
        
        # gov of candidate
        token=sent[place[0]-1] #token is gov of candidate dependency
        give_morpho(token,features,u"govtok:")
              
        # dep of candidate
        token=sent[place[1]-1]
        give_morpho(token,features,u"deptok:")

        # gov of original (if different than gov of candidate)
        if dependency[0]!=place[0]:
            token=sent[dependency[0]-1]
            give_morpho(token,features,u"Origgovtok:")
                

        # dep of original (if different than dep of candidate)
        if dependency[1]!=place[1]:
            token=sent[dependency[1]-1]
            give_morpho(token,features,u"Origdeptok:")



        #FG --- same here
        lemma=sent[place[1]-1][2]
        features.append(u"deptok:dep_token-"+lemma)
        

        ## CALCULATIONS (prefix:calc)
        
        #how many tokens between place-gov and place-dep
        #keep in mind that distancies differs depending the genre (vrt. jrc vs. hki) 
        #if place[0]>place[1]: features.append(u"calc:lengthOfJump:"+unicode(place[0]-place[1]))
        #else: features.append(u"calc:lengthOfJump:"+unicode(place[1]-place[0]))
            
        #how many items are being coordinated (how many conjs going from same place)
        
        
        ## CREATE ALL PAIRS ##
        features=set(features) #FG: so why isn't features a set to begin with? -> if there are features with weights -> dict is the way to go
        features=createAllPairs(sorted(features)) #Don't sort here

        return features


    


class RelFeatures:

    def __init__(self):
        pass
    
    # dep:(g,d), deps: GS deps deleted
    def createFeatures(self,dep,deps,sent):
        features=[]
        features.append(u"DummyFeature")
        
        
        token=sent[dep[1]-1] #token is a dep of candidate dependency (relative pronoun)
        give_morpho(token,features,u"deptok:")
        
        token=sent[dep[0]-1] #token is gov of candidate dependency
        give_morpho(token,features,u"govtok:")


        ## DEPENDENCIES (governed by deptok/govtok):
        for g,d,t in deps:
            if g==dep[1]: features.append(u"dep:DependentsDependency-"+t)
            if g==dep[0] and t!=u"rel": features.append(u"dep:GovernorsDependency-"+t)
            
        features=set(features)
        features=createAllPairs(sorted(features))
        
        return features
    


####################  GENERAL FUNCTIONS   #######################
#################################################################

#Takes features and creates all possible pairs (f1f1, f1f2, f1f3, f2f2...)
def createAllPairs(features):
    moreFeatures=[] ##FG: set
    features.sort()
    f_index=0
    #xrange
    while f_index<len(features):
        for i in xrange(f_index,len(features)):
            new_feature=features[f_index]+features[i]
            moreFeatures.append(new_feature)
        f_index+=1
    return moreFeatures

  
#create lemma and morpho features for given token
cats=set([u"CASE",u"NUM",u"SUBCAT",u"PRS",u"VOICE",u"INF"])
def give_morpho(token,f_list,prefix):
    """
    token is a one conll line
    """

    #FG: make these into sets and move it out of the function
    #cats=[u"CASE",u"NUM",u"SUBCAT",u"PRS",u"VOICE",u"INF"]

    #POSlist=[u"N",u"V",u"Adv",u"A",u"Num",u"Pron",u"Interj",u"C",u"Adp",u"Pcle",u"Punct",u"Null",u"Trash",u"Symb",u"Foreign"]
    #FeatList=[u"Nom",u"Gen",u"Par",u"Tra",u"Ess",u"Ine",u"Ela",u"Ill",u"Ade",u"Abl",u"All",u"Ins",u"Abe",u"Com"]

    lemma=token[2]
    pos=token[4]
    reading=token[6]

    f_list.append(prefix+u"Lemma-"+lemma)
    f_list.append(prefix+u"POS-"+pos)
    tags=reading.split(u"|")
    for tag in tags:
        if tag==u"_":continue
        if u"+" in tag: continue #just skip
        if u"_" in tag:
            tag=tag.split(u"_",1)[1]
#            if parts[0] in cats:
#                tag=tag.replace(parts[0]+u"_",u"",1)
#            else: continue
        f_list.append(prefix+u"Feat-"+tag)



#Writes the textfile, which has to be later converted to numbers and feed to svm.
def writeData(textfile,klass,features):
    features_str=u""
    for feature in features:
        features_str+=u" "
        features_str+=feature
    textfile.write(klass+features_str+"\n")


#Takes a sourcefile(features_textfile) and converts that to numbers
#if loadDict == True, permission to load cPickled dict from disc and use it.
#if it's False, do not load dictionary but create a new and store it to disc
def convert_toNumbers(loadDict,source,target,dir):
    if loadDict:
        f=codecs.open(dir+"/feature_num_dictionary.pkl","rb")
        featureDict=cPickle.load(f)
        f.close()
    else: featureDict={}
    sourcefile=codecs.open(source,"rt","utf-8")
    targetfile=codecs.open(target,"wt","utf-8")
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
        targetfile.write(columns[0]+u" "+features_str+"\n")
    sourcefile.close()
    targetfile.close()
    if loadDict==False:
        f=codecs.open(dir+"/feature_num_dictionary.pkl","wb")
        cPickle.dump(featureDict,f,-1)
        f.close()


