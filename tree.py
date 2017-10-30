# -*- coding: utf-8 -*-
from collections import defaultdict,namedtuple
import codecs
import sys

CoNLLFormat=namedtuple("CoNLLFormat",["ID","FORM","LEMMA","CPOS","POS","FEAT","HEAD","DEPREL","DEPS","MISC"])

#Column lists for the various formats
formats={"conll09":CoNLLFormat(0,1,2,4,4,6,8,10,12,12),"conllu":CoNLLFormat(0,1,2,3,4,5,6,7,8,9)}

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

    def __init__(self,sent,sent_id): # sent_id is for printing warnings
        self.tokens=[] #[Token(),...]
        self.childs=defaultdict(lambda:[]) #{token():[dep(),...])#
        self.govs=defaultdict(lambda:[]) #{token():[dep(),...])#
        self.basedeps={} #[deptoken():Dep()]
        self.deps=[]
        self.conjs=[] # [Dep(),...]
        self.subjs=defaultdict(lambda:[]) #{govtoken():[dep(),...])#
        self.token_dict=None
        self.from_conllu(sent,sent_id)

    def __repr__(self):
        return (" ".join(t.text for t in self.tokens)).encode(u"utf-8")

    #Called from new_from_conll() classmethod
    def from_conllu(self,lines,sent_id):    
        """ Reads conllu format and transforms it to a tree instance. """
        form=formats[u"conllu"] #named tuple with the column indices

        # create all tokens, token id is string based id from conllu!
        for line in lines:
            token=Token(line[form.ID],line[form.FORM],cpos=line[form.CPOS],pos=line[form.POS],feat=line[form.FEAT],lemma=line[form.LEMMA],misc=line[form.MISC])
            self.tokens.append(token)

        # fill in base layer syntax
        for line in lines:

            dep_token=self.get_token(line[form.ID])

            if line[form.HEAD]==u"0": # sentence root
                dep_token.root=True
                continue

            if line[form.HEAD]==u"_": # null or multiword expression, skip
                continue
            
            base_dependency=Dep(self.get_token(line[form.HEAD]),dep_token,line[form.DEPREL],flag=u"BASE")
            self.add_dep(base_dependency)

        # fill in other dependencies
        for line in lines: # search for other incoming dependencies
            if line[form.DEPS]!=u"_":
                dep_token=self.get_token(line[form.ID])
                for dep in line[form.DEPS].split(u"|"): # all additional dependencies
                    g,t=dep.split(u":",1)
                    gov_token=self.get_token(g)
                    flag=None
                    if t==u"flat:name":
                        flag=u"NAME"
                    if "." in g or "." in line[form.ID]:
                        flag="NULLDEP"
                    if not flag:
                        for conj in self.conjs:
                            if conj.dep==dep_token or conj.dep==gov_token:
                                flag=u"CC"
                                break
                    if not flag and (t==u"nsubj" or t==u"nsubj:cop"):
                        flag=u"XS"
                    if not flag:
                        flag=u"UNREC"
                        print >> sys.stderr, "warning! unrecognized dependency:",sent_id,line[form.ID],line[form.HEAD],line[form.DEPREL],line[form.DEPS]
                        #continue
                    dependency=Dep(gov_token,dep_token,t,flag=flag)
                    self.add_dep(dependency)

    def get_token(self,tid):
        if not self.token_dict:
            self.token_dict={}
            for i,token in enumerate(self.tokens):
                self.token_dict[token.index]=i
        return self.tokens[self.token_dict[tid]]            

    def add_dep(self,dependency):
        self.deps.append(dependency)
        if dependency.flag==u"BASE":
            self.basedeps[dependency.dep]=dependency
        self.childs[dependency.gov].append(dependency)
        self.govs[dependency.dep].append(dependency)
        if dependency.dtype==u"conj":
            self.conjs.append(dependency)
        elif dependency.dtype==u"nsubj" or dependency.dtype==u"nsubj:cop":
            self.subjs[dependency.gov].append(dependency)

    def has_dep(self,g,d):
        for dependency in self.deps:
            if dependency.gov.index==g.index and dependency.dep.index==d.index:
                return dependency.dtype
        return None


    def tree_to_conll(self,conll_format=u"conll09"):
        print >> sys.stderr, "CONLL09 NOT SUPPORTED!!!"
        sys.exit()
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

        
    def to_conllu(self,no_l2=False):
        form=formats[u"conllu"]
        sent=[]
        for token in self.tokens:
            line=[]
            for _ in xrange(10):
                line.append(u"_")
            line[form.ID]=token.index
            line[form.FORM]=token.text
            line[form.LEMMA]=token.lemma
            line[form.CPOS]=token.cpos
            line[form.POS]=token.pos
            line[form.FEAT]=token.feat
            baselayer=self.basedeps.get(token,None)
            if baselayer is None:
                if token.root==True:
                    line[form.HEAD]=u"0"
                    line[form.DEPREL]=u"root"
                else:
                    line[form.HEAD]=u"_"
                    line[form.DEPREL]=u"_"
            else:
                line[form.HEAD]=baselayer.gov.index
                line[form.DEPREL]=baselayer.dtype
            extradeps=[]
            for d in self.govs.get(token,[]): # d is dep object
                if d!=baselayer:
                    if no_l2==True and (d.flag=="CC" or d.flag=="XS"):
                        continue
                    extradeps.append((d.gov.index,d.dtype.split(u"&",1)[-1]))
            if not extradeps:
                 line[form.DEPS]=u"_"
            else:
                line[form.DEPS]=u"|".join(d[0]+u":"+d[1] for d in sorted(extradeps, key=lambda x: (float(x[0]),x[1].lower())))
            line[form.MISC]=token.misc
            sent.append(line)
        for line in sent:
            print >> sys.stdout, (u"\t".join(c for c in line)).encode(u"utf-8")
        print >> sys.stdout  


class Token(object):

    def __init__(self,idx,text,cpos=u"_",pos="_",feat="_",lemma="_",misc="_"):
        self.index=idx
        self.text=text
        self.cpos=cpos
        self.pos=pos
        self.feat=feat
        self.lemma=lemma
        self.misc=misc
        self.root=False

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

    def __hash__(self):
        return hash(self.gov.text+u"--"+self.dep.text+u"--"+self.dtype+u"--"+self.flag)


    

    

