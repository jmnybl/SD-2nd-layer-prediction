# remove second layer dependencies from conllu file
from tree import Tree, read_conll
import sys
import codecs

count=0
for comments,sent in read_conll(codecs.getreader(encoding="utf-8")(sys.stdin)):
    for comm in comments:
        print >> sys.stdout, comm.encode(u"utf-8")
    t=Tree(sent,count+1)
    t.to_conllu(no_l2=True)
    count+=1
    
