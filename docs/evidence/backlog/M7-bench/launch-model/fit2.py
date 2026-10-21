from model2 import *
import itertools
gf=list(itertools.product([0.5,1,2,3],[0,5,10,20,30,50],[0.1,0.25,0.5],[0,10,30,50],[0.5,1],[30,100,300]))
gb=list(itertools.product([0.5,1,2,3],[0,5,10,20,30,50],[0.1,0.25,0.5],[0,10,30,50],[0.5,1],[30,100,300]))
for name,table,choose,grid in [('forward',F,fwd_choice,gf),('backward',B,bwd_choice,gb)]:
    res=[]
    for *prm,l in grid:
        s=summ(regret(table,choose,tuple(prm),l)); res.append((s,tuple(prm),l))
    res.sort(key=lambda r:(r[0][2],r[0][0])); print(name,'excess',res[:5])
    res.sort(key=lambda r:(r[0][0],r[0][2])); print(name,'mean',res[:3])
