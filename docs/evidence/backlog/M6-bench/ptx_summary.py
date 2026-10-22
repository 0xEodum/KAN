# Backlog M6: register, stack and spill use of the new kernels from a scratch build with
# -DCMAKE_CUDA_ARCHITECTURES=<arch> -DCMAKE_CUDA_FLAGS="-Xptxas -v" (build log as argument).
import re,sys
# ptxas -v: "Compiling entry function '<mangled>' for 'sm_XX'" then "Function properties ... N bytes stack frame, S bytes spill stores, L bytes spill loads" then "Used R registers"
for path in sys.argv[1:]:
    text=open(path).read().splitlines()
    cur=None; props=None
    out=[]
    for line in text:
        m=re.search(r"Compiling entry function '(\S+)' for '(sm_\d+)'",line) or re.search(r"Function properties for (\S+)",line)
        if 'Compiling entry function' in line:
            m=re.search(r"entry function '(\S+)'",line); cur=m.group(1)
        m2=re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",line)
        if m2 and cur: props=m2.groups()
        m3=re.search(r"Used (\d+) registers",line)
        if m3 and cur and props:
            if re.search(r"refit|grid_|GridSpline|basis_grid|spline_grid",cur):
                out.append((cur,m3.group(1),*props))
            cur=None; props=None
    print(path)
    for c,r,st,ss,sl in sorted(set(out)):
        name=re.sub(r'_ZN3kan4cuda\w*?_GLOBAL__N__\w+?_11_resident_cu_\w{8}','',c)
        print(f"  regs {r:>3} stack {st:>4} spill {ss}/{sl}  {name[:120]}")
