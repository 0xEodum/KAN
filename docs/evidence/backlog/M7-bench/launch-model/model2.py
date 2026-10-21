import math, json, collections, itertools
from model import RES, chunks, plan, load, GAP
P=48
def cost(batch,S,groups,slots,it,R,prm):
    ov,lat,g,pb,pf=prm
    span=256*S; c=chunks(batch,S); last=batch-(c-1)*span
    act=math.ceil(last/256)
    slots=max(1,min(slots,c))
    blocks=groups*slots
    cpb=c/slots  # chunks per block (average)
    seff=((c-1)*S+act+pf*(S-act))/c
    per=blocks/P; n=min(math.ceil(per),R)
    tb=cpb*it*max(n*(seff+ov),lat)+pb*n
    return (max(1.0,per/n)+g)*tb
def fwd_choice(prec,i,o,K,b,prm,launch,cands):
    best=None
    for (S,z) in sorted(cands,key=lambda t:(-t[0],t[1])):
        R=RES[prec]['fwd'][S]; c=chunks(b,S); wave=P*R
        most=1 if (i<2 or o*c>=wave) else min(i,wave//(o*c))
        if z>most: continue
        per=-(-i//z)
        v=cost(b,S,o*z,c,per*K,R,prm)+(launch if z>1 else 0)
        if best is None or v<best[0]: best=(v,(S,z))
    return best[1]
def bwd_choice(prec,i,o,K,b,prm,launch,cands,maxsplit=16):
    slots,_=plan(prec,i,o,K,b)
    best=None
    for (S,z) in sorted(cands,key=lambda t:(-t[0],t[1])):
        if z>min(maxsplit,o): continue
        R=RES[prec]['bwd'][S]
        per=-(-o//z)
        v=cost(b,S,i*z,min(slots,chunks(b,S)),per*K,R,prm)+(launch if z>1 else 0)
        if best is None or v<best[0]: best=(v,(S,z))
    return best[1]
F,B=load('kf32.json'); F64,B64=load('kf64.json'); F.update(F64); B.update(B64)
def regret(table,choose,prm,launch):
    rs=[]
    for key,m in table.items():
        p,i,o,K,b=key
        ch=choose(p,i,o,K,b,prm,launch,list(m.keys()))
        rs.append((m[ch]/min(m.values()),key,ch,min(m,key=m.get),m[ch]-min(m.values()),min(m.values())))
    return rs
def summ(rs):
    v=[r[0] for r in rs]; ex=sum(r[4] for r in rs)/sum(r[5] for r in rs)
    return round(sum(v)/len(v),4), round(max(v),3), round(ex,4)
if __name__=='__main__':
    grid=list(itertools.product([0.5,1,2,3],[0,5,10,20,30,50],[0,0.25,0.5,1],[0,10,30,100]))
    for name,table,choose in [('forward',F,fwd_choice),('backward',B,bwd_choice)]:
        res=[]
        for prm in grid:
            for launch in [0,10,30,100,300]:
                s=summ(regret(table,choose,prm,launch)); res.append((s,prm,launch))
        res.sort(key=lambda r:r[0][0]); print(name,'mean',res[:4])
        res.sort(key=lambda r:r[0][2]); print(name,'excess',res[:4])
