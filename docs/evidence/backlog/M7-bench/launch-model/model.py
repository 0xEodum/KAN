import math, json, collections, itertools
P=48
RES={'f32':{'fwd':{1:6,2:5,4:6,8:3},'bwd':{1:4,2:3,4:3,8:3}},
     'f64':{'fwd':{1:5},'bwd':{1:2,2:2,4:1}}}
GAP=1.0  # us per extra kernel in a graph
def chunks(b,S): return -(-b//(256*S))
PF=0.5
def cost(batch,S,groups,slots,it,P,R,ov,lat):
    # busiest multiprocessor: waves of at most R co-resident blocks; each
    # iteration (one wavelet) of a wave takes max(issue = n (S_eff + ov), lat)
    span=256*S; c=chunks(batch,S); last=batch-(c-1)*span
    act=math.ceil(last/256)
    s_part=act+PF*(S-act)
    slots=max(1,min(slots,c))
    blocks=groups*slots
    # iterations of the busiest block and the average slot cost
    per_block_chunks=math.ceil(c/slots)
    s_avg=((c-1)*S+s_part)/c
    per=math.ceil(blocks/P); n=min(per,R); waves=math.ceil(per/R)
    seff = s_part if c==1 else s_avg
    return waves*per_block_chunks*it*max(n*(seff+ov), lat)
def plan(prec,i,o,K,cap):
    Rb=RES[prec]['bwd']; Smax=max(Rb); wave=P*Rb[Smax]
    count=i*o*K
    most=max(1,chunks(cap,1)); budget=max(1,cap*(i+o)//max(1,3*count))
    slots=max(chunks(cap,2 if prec=='f64' else 4), (wave-1)//i+1)
    slots=max(1,min(slots,most,budget,65535))
    splits=1 if i*slots>=wave else min((wave-1)//(i*slots)+1,o,16)
    return slots,splits
def fwd_choice(prec,i,o,K,b,ov,sat,launch,cands):
    best=None
    for (S,z) in sorted(cands,key=lambda t:(-t[0],t[1])):
        R=RES[prec]['fwd'][S]; c=chunks(b,S); wave=P*R
        most=1 if (i<2 or o*c>=wave) else min(i,wave//(o*c))
        if z>most: continue
        per=-(-i//z)
        v=cost(b,S,o*z,c,per*K,P,R,ov,sat)+(launch if z>1 else 0)
        if best is None or v<best[0]: best=(v,(S,z))
    return best[1]
def bwd_choice(prec,i,o,K,b,ov,sat,launch,cands):
    slots,maxz=plan(prec,i,o,K,b)
    best=None
    for (S,z) in sorted(cands,key=lambda t:(-t[0],t[1])):
        if z>max(1,min(maxz,o)): continue
        R=RES[prec]['bwd'][S]
        per=-(-o//z)
        v=cost(b,S,i*z,min(slots,chunks(b,S)),per*K,P,R,ov,sat)+(launch if z>1 else 0)
        if best is None or v<best[0]: best=(v,(S,z))
    return best[1]
def load(path):
    d=json.load(open(path))
    F=collections.defaultdict(dict); B=collections.defaultdict(dict)
    for line,v in d.items():
        p,i,o,K,b,fs,bs,sl,sp,isp=line.split(); i,o,K,b,fs,bs,sp,isp=map(int,(i,o,K,b,fs,bs,sp,isp))
        key=(p,i,o,K,b)
        F[key][(fs,isp)]=v['fwd']+GAP*v['nfwd']
        B[key][(bs,sp)]=v['bwd']+GAP*v['nbwd']
    return F,B
