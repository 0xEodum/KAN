import torch, time
def cheb(x, K):
    T=[torch.ones_like(x), x]
    for _ in range(K-2): T.append(2*x*T[-1]-T[-2])
    return torch.stack(T,-1)            # B,I,K
def step(params, x, up, K, lr=1e-3):
    h=x
    for C,b in params:                  # C: O, I*K
        h = cheb(h,K).flatten(1) @ C.T + b   # KAN layer == basis map + GEMM
    h.backward(up)
    with torch.no_grad():
        for C,b in params: C-=lr*C.grad; b-=lr*b.grad; C.grad=None; b.grad=None
def bench(dims,B,K,dt,reps=50):
    torch.manual_seed(0)
    params=[(torch.randn(o,i*K,device='cuda',dtype=dt).mul_(0.1/(i*K)**.5).requires_grad_(),
             torch.zeros(o,device='cuda',dtype=dt).requires_grad_()) for i,o in zip(dims,dims[1:])]
    x=torch.rand(B,dims[0],device='cuda',dtype=dt)*2-1; up=torch.randn(B,dims[-1],device='cuda',dtype=dt)/B
    for _ in range(5): step(params,x,up,K)
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(reps): step(params,x,up,K)
    torch.cuda.synchronize(); return (time.perf_counter()-t)/reps*1e3
for dims,B in [((64,64,32,16),1024),((16,24,8),1024),((256,256,256,10),8192),((1024,1024,1024),4096)]:
    print(dims,B, " fp64 %.3f ms  fp32 %.3f ms"%(bench(dims,B,7,torch.float64),bench(dims,B,7,torch.float32)))
