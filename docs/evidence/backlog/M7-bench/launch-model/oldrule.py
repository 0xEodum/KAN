from model2 import *
def bs(batch,most):
    s=1
    while s<most and s*2*256*15<=16*batch: s*=2
    return s
def old_f(prec,i,o,K,b,prm,launch,cands):
    S=bs(b,1 if prec=='f64' else 8)
    if S==8 and o*chunks(b,8)<144: S=4
    return (S,1)
def old_b(prec,i,o,K,b,prm,launch,cands):
    wave=144 if prec=='f32' else 48
    S=bs(b,4 if prec=='f64' else 8)
    while S>2 and 2*i*chunks(b,S)<wave: S//=2
    slots,splits=plan(prec,i,o,K,b)
    sl=min(chunks(b,S),slots)
    z=1
    if i*sl<wave: z=min(splits,(wave-1)//(i*sl)+1)
    per=-(-o//z); z=-(-o//per)
    if (S,z) not in cands:
        # nearest measured z
        zs=[c[1] for c in cands if c[0]==S]; z=min(zs,key=lambda t:abs(t-z))
    return (S,z)
if __name__=="__main__":
  for name,table,choose in [("forward",F,old_f),('backward',B,old_b)]:
      print(name,'old rule',summ(regret(table,choose,None,0)))
  for name,table,choose,prm,l in [('forward',F,fwd_choice,(2,20,0.25,30),100),('backward',B,bwd_choice,(1,50,0.25,30),300)]:
      print(name,'model',summ(regret(table,choose,prm,l)))
