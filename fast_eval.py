import codecs
import argparse


def evaluate(args):
    gsf=codecs.open(args.gs,u"rt",u"utf-8")
    newf=codecs.open(args.sys,u"rt",u"utf-8")

    tp=0
    fp=0
    fn=0

    for gsl,newl in zip(gsf,newf):
        gsl=gsl.strip()
        newl=newl.strip()
        if not gsl: continue
        # 10: deprel, 8: head
        gs=[]
        for g,d in zip(gsl.split(u"\t")[8].split(u","),gsl.split(u"\t")[10].split(u",")):
            gs.append((g,d))
        sys=[]
        for g,d in zip(newl.split(u"\t")[8].split(u","),newl.split(u"\t")[10].split(u",")):
            sys.append((g,d))
        if len(gs)==1 and len(sys)==1: # no second layer dependencies
            continue
        del gs[0] # remove the first layer deps
        del sys[0]
        for dep in gs:
            if dep in sys:
                tp+=1
            else:
                fn+=1
        for dep in sys:
            if dep not in gs:
                fp+=1


    gsf.close()
    newf.close()

    pre=tp/float(tp+fp)
    rec=tp/float(tp+fn)

    f=2*pre*rec/(pre+rec)

    print "pre",pre
    print "rec",rec
    print "f",f
    print

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate second layer predictions.')
    g=parser.add_argument_group("Input")
    g.add_argument('--gs', required=True, dest='gs', help='Gold standard file.')
    g.add_argument('--sys', required=True, dest='sys', help='System predictions.')

    args = parser.parse_args()
    
    evaluate(args)
