import codecs
import argparse

DEPS=8

def evaluate(args):
    gsf=codecs.open(args.gs,u"rt",u"utf-8")
    newf=codecs.open(args.sys,u"rt",u"utf-8")

    tp=0
    fp=0
    fn=0 

    for gsl,newl in zip(gsf,newf):
        gsl=gsl.strip()
        newl=newl.strip()
        if not gsl or gsl.startswith(u"#"): continue
        
        gs=[]
        for dep in gsl.split(u"\t")[DEPS].split(u"|"):
            if (dep!=u"_") and (u"name" not in gsl): # TODO: what to do with propagated names?
                gs.append(dep)
        sys=[]
        for dep in newl.split(u"\t")[DEPS].split(u"|"):
            if (dep!=u"_") and (u"name" not in newl):
                sys.append(dep)
        if len(gs)==0 and len(sys)==0: # no second layer dependencies
            continue

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
