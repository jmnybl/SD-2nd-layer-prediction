# -*- coding: utf-8 -*-
from collections import defaultdict,namedtuple
import codecs
import sys

CoNLLFormat=namedtuple("CoNLLFormat",["ID","FORM","LEMMA","POS","FEAT","HEAD","DEPREL"])

#Column lists for the various formats
formats={"conll09":CoNLLFormat(0,1,2,4,6,8,10)}

def read_conll(inp):
    """ Read conll format file and yield one sentence at a time as a list of lists of columns. If inp is a string it will be interpreted as filename, otherwise as open file for reading in unicode"""
    if isinstance(inp,basestring):
        f=codecs.open(inp,u"rt",u"utf-8")
    elif inp==None: #Do stdin
        f=codecs.getreader("utf-8")(sys.stdin)
    else:
        f=inp
    
    comments=[]
    sent=[]
    for line in f:
        line=line.strip()
        if not line: # new sentence
            if sent:
                yield comments,sent
            sent=[]
            comments=[]
        elif line.startswith(u"#"): # comment
            comments.append(line)
        else: # normal line
            sent.append(line.split(u"\t"))
    else:
        if sent:
            yield comments,sent

    if isinstance(inp,basestring):
        f.close() #Close it if you opened it

class Tree(object):

    def __init__(self,sent):
        self.tokens=[] #[Token(),...]
        self.childs=defaultdict(lambda:[]) #{token():[dep(),...])#
        self.govs=defaultdict(lambda:[]) #{token():[dep(),...])#
        self.basedeps={} #[deptoken():Dep()]
        self.deps=[]
        self.conjs=[] # [Dep(),...]
        self.rels=[] # [Dep(),...]
        self.subjs=defaultdict(lambda:[]) #{govtoken():[dep(),...])#
        self.from_conll(sent)

    #Called from new_from_conll() classmethod
    def from_conll(self,lines,conll_format="conll09"):    
        """ Reads conll format and transforms it to a tree instance. `conll_format` is a format name
            which will be looked up in the formats module-level dictionary"""
        form=formats[conll_format] #named tuple with the column indices
        for i in xrange(0,len(lines)): # create tokens
            line=lines[i]
            token=Token(i,line[form.FORM],pos=line[form.POS],feat=line[form.FEAT],lemma=line[form.LEMMA])
            self.tokens.append(token)

        # fill syntax
        for line in lines:
            govs=line[form.HEAD].split(u",")
            dep=self.tokens[int(line[0])-1]
            if len(govs)==1:
                if int(govs[0])==0:
                    continue
                gov=self.tokens[int(govs[0])-1]
                dtype=line[form.DEPREL]
                dependency=Dep(gov,dep,dtype,flag=u"BASE")
                self.add_dep(dependency)
            else:
                deprels=line[form.DEPREL].split(u",")
                for idx,(gov,dtype) in enumerate(zip(govs,deprels)):
                    if int(gov)==0:
                        continue
                    gov_token=self.tokens[int(gov)-1]
                    if idx==0:
                        dependency=Dep(gov_token,dep,dtype,flag=u"BASE")
                    else:
                        if (u"rel" in deprels) and dtype!=u"rel":
                            for gov2,dtype2 in zip(govs,deprels):
                                if dtype2==u"rel" and gov2==gov:
                                    dtype=u"rel&"+dtype
                                    break
                        if (dtype==u"xsubj" or dtype==u"xsubj-cop") and (u"conj" not in deprels): # TODO: jumped xsubj?
                            flag=u"XS"
                        elif u"&" in dtype: # TODO: jumped rel
                            flag=u"REL"
                        else:
                            flag=u"CC"
                        dependency=Dep(gov_token,dep,dtype,flag)
                    self.add_dep(dependency)
            

    def add_dep(self,dependency):
        self.deps.append(dependency)
        if dependency.flag==u"BASE":
            self.basedeps[dependency.dep]=dependency
        self.childs[dependency.gov].append(dependency)
        self.govs[dependency.dep].append(dependency)
        if dependency.dtype==u"rel":
            self.rels.append(dependency)
        elif dependency.dtype==u"conj":
            self.conjs.append(dependency)
        elif dependency.dtype==u"nsubj" or dependency.dtype==u"nsubj-cop":
            self.subjs[dependency.gov].append(dependency)

    def has_dep(self,g,d):
        for dependency in self.deps:
            if dependency.gov.index==g.index and dependency.dep.index==d.index:
                return dependency.dtype
        return None


    def tree_to_conll(self,conll_format=u"conll09"):
        #["ID","FORM","LEMMA","POS","FEAT","HEAD","DEPREL"]
        form=formats[conll_format]
        sent=[]
        for token in self.tokens:
            line=[]
            for i in xrange(14):
                line.append(u"_")
            line[form.ID]=str(token.index+1)
            line[form.FORM]=token.text
            line[form.LEMMA]=token.lemma
            line[form.LEMMA+1]=token.lemma # repeat everything...
            line[form.POS]=token.pos
            line[form.POS+1]=token.pos
            line[form.FEAT]=token.feat
            line[form.FEAT+1]=token.feat
            if not self.govs[token]:
                govs=u"0"
                deprels=u"ROOT"
            else:
                govs=u",".join(str(dep.gov.index+1) for dep in self.govs[token])
                deprels=u",".join(dep.dtype.split(u"&",1)[-1] for dep in self.govs[token])
            line[form.HEAD]=govs
            line[form.HEAD+1]=govs
            line[form.DEPREL]=deprels
            line[form.DEPREL+1]=deprels
            sent.append(line)
        for line in sent:
            print >> sys.stdout, (u"\t".join(c for c in line)).encode(u"utf-8")
        print >> sys.stdout    

        
    def to_conllu(self):
        #["ID","FORM","LEMMA","POS","FEAT","HEAD","DEPREL"]
        ID,FORM,LEMMA,CPOS,POS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)
        sent=[]
        for token in self.tokens:
            line=[]
            for _ in xrange(10):
                line.append(u"_")
            line[ID]=str(token.index+1)
            line[FORM]=token.text
            line[LEMMA]=token.lemma
            line[CPOS]=token.pos
            line[POS]=token.pos # TODO: full conllu support
            line[FEAT]=token.feat # TODO: full conllu support
            baselayer=self.basedeps.get(token,None)
            if baselayer is None:
                line[HEAD]=u"0"
                line[DEPREL]=u"ROOT"
            else:
                line[HEAD]=str(baselayer.gov.index+1)
                line[DEPREL]=baselayer.dtype
            extradeps=[]
            for d in self.govs.get(token,[]):
                if d!=baselayer:
                    extradeps.append((d.gov.index+1,d.dtype.split(u"&",1)[-1]))   
            line[DEPS]=u"|".join(str(d[0])+u":"+d[1] for d in sorted(extradeps)) if extradeps else u"_"
            sent.append(line)
        for line in sent:
            print >> sys.stdout, (u"\t".join(c for c in line)).encode(u"utf-8")
        print >> sys.stdout  


class Token(object):

    def __init__(self,idx,text,pos="",feat="",lemma=""):
        self.index=idx
        self.text=text
        self.pos=pos
        self.feat=feat
        self.lemma=lemma

    def __str__(self):
        return self.text.encode(u"utf-8")

    def __repr__(self):
        return (u"<"+self.text+u">").encode(u"utf-8")

    def __eq__(self,other):
        return self.index==other.index

    def __hash__(self):
        return hash(self.index)



class Dep(object):

    def __init__(self,gov,dep,dtype,flag=None):
        self.gov=gov
        self.dep=dep
        self.dtype=dtype
        self.flag=flag

    def __eq__(self,other):
        return self.gov==other.gov and self.dep==other.dep and self.dtype==other.dtype and self.flag==other.flag

    def __str__(self):
        return (self.gov.text+u"--"+self.dep.text+u"--"+self.dtype+u"--"+self.flag).encode(u"utf-8")

    def __repr__(self):
        return (self.gov.text+u"--"+self.dep.text+u"--"+self.dtype+u"--"+self.flag).encode(u"utf-8")


    

    

