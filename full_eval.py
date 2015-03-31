from tree import read_conll,Tree
import codecs
import sys
import argparse

def print_numbers(title,stats):

    print title
    print stats[0],stats[1],stats[2]
    try:
        pre=stats[0]/float(stats[0]+stats[1])
        rec=stats[0]/float(stats[0]+stats[2])

        f=2*pre*rec/(pre+rec)
    except:
        pre,rec,f=0.0,0.0,0.0

    print "pre",pre
    print "rec",rec
    print "f",f
    print
    

def evaluate(args):    

    jump_stats=[0,0,0] # tp,fp,fn
    xsubj_stats=[0,0,0]

    for (comm_s,sent_s),(comm_g,sent_g) in zip(read_conll(args.sys),read_conll(args.gs)):

        tree_s=Tree(sent_s)
        tree_g=Tree(sent_g)

        jump_s=set([d for d in tree_s.deps if d.flag==u"CC"])
        xsubj_s=set([d for d in tree_s.deps if d.flag==u"XS"])

        jump_g=set([d for d in tree_g.deps if d.flag==u"CC"])
        xsubj_g=set([d for d in tree_g.deps if d.flag==u"XS"])

        jump_stats[0]+=len(jump_s&jump_g)
        jump_stats[1]+=len(jump_s-jump_g)
        jump_stats[2]+=len(jump_g-jump_s)

        xsubj_stats[0]+=len(xsubj_s&xsubj_g)
        xsubj_stats[1]+=len(xsubj_s-xsubj_g)
        xsubj_stats[2]+=len(xsubj_g-xsubj_s)

    print_numbers("Conjunct propagation:",jump_stats)
    print_numbers("External subjects:",xsubj_stats)
    total_stats=[0,0,0]
    for i in range(0,len(jump_stats)):
        total_stats[i]=jump_stats[i]+xsubj_stats[i]
    print_numbers("Total:",total_stats)
    
    


if __name__==u"__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate second layer predictions.')
    g=parser.add_argument_group("Input")
    g.add_argument('--gs', required=True, dest='gs', help='Gold standard file.')
    g.add_argument('--sys', required=True, dest='sys', help='System predictions.')

    args = parser.parse_args()

    print "Gold standard:",args.gs
    print "System output:",args.sys

    evaluate(args)
