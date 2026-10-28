"""usage: python compare.py OPT0.log OPT3.log -- compare the actions the dolls chose per decision in two brainlevel logs."""
import re,sys
def rows(p):
    r=[]
    for l in open(p):
        m=re.match(r"decision\s+(\d+)/\d+\s+sees (.*?)\s+his dolls chose (.*?) \(logits L/R/F (\[.*?\])",l)
        if m: r.append((m.group(2),m.group(3),m.group(4)))
    return r
if len(sys.argv) != 3:
    sys.exit(__doc__)
a,b=rows(sys.argv[1]),rows(sys.argv[2])
print(len(a),len(b))
d={}
bad=0
for s,act,lg in a+b:
    k=s
    if k in d and d[k]!=(act,lg): bad+=1; print("DIFF",k,d[k],(act,lg))
    d.setdefault(k,(act,lg))
print("distinct token contexts",len(d),"conflicts",bad)
sa={s for s,_,_ in a}; sb={s for s,_,_ in b}; print("contexts in both",len(sa&sb))
print("same sequence of actions:",[x[1] for x in a]==[x[1] for x in b], sum(x[1]==y[1] for x,y in zip(a,b)),"of",min(len(a),len(b)),"decisions at the same index chose the same action")
from collections import Counter
print(Counter(x[1] for x in a),Counter(x[1] for x in b))
