import sys, numpy as np
sys.path.insert(0,sys.argv[1])
import kan

def config(m,n,center=.13,scale=1.7,epsilon=1e-8):
    c=kan.RationalConfig();c.numerator_degree=m;c.denominator_degree=n;c.center=center;c.scale=scale;c.epsilon=epsilon
    return c

def near(a,b,rtol=2e-11,atol=2e-13):
    np.testing.assert_allclose(a,b,rtol=rtol,atol=atol)

def independent_scalar():
    count=0
    for m in range(17):
      for n in range(17):
        c=config(m,n);a=np.sin(np.arange(m+1)+.7)*.13;b=np.cos(np.arange(n)+.31)*.004
        for x in [-2.17,-.27,.13,.99,2.19]:
          z=(x-c.center)/c.scale;pa=z**np.arange(m+1);pb=z**np.arange(1,n+1)
          p=a@pa;q=1+b@pb;dp=0 if m==0 else (a[1:]*np.arange(1,m+1))@pa[:-1]
          dq=0 if n==0 else (b*np.arange(1,n+1))@(z**np.arange(n))
          v,dx,da,db=kan.evaluate_rational(c,x,a,b)
          near(v,p/q);near(dx,(dp*q-p*dq)/(q*q*c.scale));near(da,pa/q);near(db,-p*pb/(q*q))
          count+=1
    print('PASS independent scalar identities, degree pairs 0..16:',count)

def strict():
    c=config(1,1);a=np.array([.3,.2]);b=np.array([.1])
    bads=[a.astype(np.float32),a.astype('>f8'),a.reshape(1,2),np.arange(4,dtype=float)[::2],np.array([np.inf,.2])]
    for bad in bads:
      try: kan.evaluate_rational(c,.2,bad,b)
      except (TypeError,ValueError): pass
      else: raise AssertionError('accepted wrong dtype, shape, stride or nonfinite')
    # Deliberately unaligned float64 array.
    bad=np.ndarray((2,),dtype=np.float64,buffer=bytearray(17),offset=1)
    try: kan.evaluate_rational(c,.2,bad,b)
    except ValueError: pass
    else: raise AssertionError('accepted unaligned array')
    print('PASS strict scalar array interfaces')

def mixed():
    r=kan.Layer(2,3,config(4,3));r.set_rational_parameters(np.sin(np.arange(30)).reshape(3,2,5)*.06,np.cos(np.arange(18)).reshape(3,2,3)*.015,np.array([.01,-.02,.04]))
    c=kan.BasisConfig();c.kind=kan.BasisKind.GaussianRbf;c.size=3;c.trainable_rbf=True;c.centers=[-.7,0,.8];c.log_widths=[-.1,.2,.4]
    b=kan.Layer(3,2,c);b.set_parameters(np.sin(np.arange(18)).reshape(2,3,3)*.12,np.array([-.02,.03]))
    q=kan.Layer(2,2,config(1,2));q.set_rational_parameters(np.array([.1,.3,-.2,.4,.1,-.7,.2,-.5]).reshape(2,2,2),np.array([.05,.1,-.07,.02,.09,-.1,.01,.03]).reshape(2,2,2),np.array([.01,.02]))
    network=kan.Network([r,b,q]);x=np.array([[-.73,.41],[.17,-.21],[.88,-.66]]);u=np.array([[.31,-.22],[-.48,.2],[.17,.12]])
    g=network.backward(x,u);h=1e-6
    def objective(n,v=x):return float((n.forward(v)*u).sum())
    for j in np.ndindex(x.shape):
      p=x.copy();m=x.copy();p[j]+=h;m[j]-=h;near(g.input[j],(objective(network,p)-objective(network,m))/(2*h),atol=1e-9)
    checks=0
    for index,l in enumerate(network.layers):
      names=['coefficients','denominators','bias'] if l.is_rational else ['coefficients','bias','centers','log_widths']
      for name in names:
        initial=np.array(getattr(l.basis,name) if name in ['centers','log_widths'] else getattr(l,name))
        for j in np.ndindex(initial.shape):
          results=[]
          for sign in [1,-1]:
            ls=network.layers;t=ls[index];values=initial.copy();values[j]+=sign*h
            if name in ['centers','log_widths']:
              cen=np.array(t.basis.centers);lw=np.array(t.basis.log_widths)
              t.set_rbf_parameters(values if name=='centers' else cen,values if name=='log_widths' else lw)
            else:
              a=t.coefficients;v=t.bias
              if name=='coefficients':a=values
              if name=='bias':v=values
              if t.is_rational:t.set_rational_parameters(a,values if name=='denominators' else t.denominators,v)
              else:t.set_parameters(a,v)
            results.append(objective(kan.Network(ls)))
          near(getattr(g.layers[index],name)[j],(results[0]-results[1])/(2*h),atol=1e-9);checks+=1
    # Networks/layers/config/gradients are owned snapshots.
    snapshot=network.layers;snapshot[0].set_rational_parameters(snapshot[0].coefficients*99,snapshot[0].denominators,snapshot[0].bias)
    near(network.forward(x),kan.Network([r,b,q]).forward(x),atol=0,rtol=0)
    if kan.cuda_available():
      gpu=kan.ResidentNetwork(network,5);gpu.upload_input(x);gpu.upload_output_gradient(u);alloc=gpu.workspace_allocations
      for step in range(5):
        gpu.forward();near(gpu.download_output(),network.forward(x));gpu.backward(.13);gg=gpu.download_gradients();cg=network.backward(x,u);_,rg=network.regularization(.13)
        for i in range(3):
          for name in ['input','denominators','bias','centers','log_widths']:near(getattr(gg.layers[i],name),getattr(cg.layers[i],name))
          near(gg.layers[i].coefficients,cg.layers[i].coefficients+rg.layers[i].coefficients)
        # CPU does not expose mutable gradient objects; independent GPU trajectory is
        # compared with loss-only updates below; separate L2 exact checks above suffice.
        gpu.backward();gg=gpu.download_gradients();network.sgd(cg,.03);gpu.sgd(.03)
        for l,gl in zip(network.layers,gpu.download_parameters().layers):
          near(gl.coefficients,l.coefficients);near(gl.denominators,l.denominators);near(gl.bias,l.bias)
          if not l.is_rational:near(gl.basis.centers,l.basis.centers);near(gl.basis.log_widths,l.basis.log_widths)
      assert alloc==gpu.workspace_allocations
    print('PASS mixed rational/trainable-RBF/rational input and',checks,'parameter finite differences and snapshots; GPU five-step trajectory:',kan.cuda_available())

def failures():
    for n,x in [(2,1e-200),(16,1e-20),(16,1e-30),(15,-1e-30)]:
      c=config(0,n,0,1);v,dx,da,db=kan.evaluate_rational(c,x,np.array([1e300]),np.zeros(n))
      # Compute each reference product in logarithms, avoiding the very underflow
      # which is being challenged in the numerical implementation.
      expected=np.array([-((-1)**k if x<0 else 1)*np.exp(np.log(1e300)+k*np.log(abs(x))) for k in range(1,n+1)])
      near(db,expected,rtol=3e-12,atol=0)
    print('PASS representable denominator VJPs through zero/subnormal powers, unequal order and negative z')
    # High dynamic range finite quotient; any declared intermediate overflow may
    # explicitly fail, but successful results must retain nonzero finite VJPs.
    c=config(0,1,0,1);a=np.array([1e300]);b=np.array([1e200])
    try:
      v,dx,da,db=kan.evaluate_rational(c,1,a,b);near(v,1e100,atol=0);near(dx,-1e100,atol=0);near(db,[-1e-100],atol=0)
      print('PASS high dynamic range finite quotient/VJPs')
    except OverflowError:print('PASS high dynamic range explicitly rejected intermediate overflow')
    if not kan.cuda_available():return
    # A forward-safe sample whose denominator VJP overflows only in backward.
    c=config(0,1,0,1);l=kan.Layer(1,1,c);l.set_rational_parameters(np.array([[[1e300]]]),np.array([[[0.]]]),np.zeros(1))
    gpu=kan.ResidentNetwork(kan.Network([l]),1);gpu.upload_input(np.array([[1.]]));gpu.upload_output_gradient(np.array([[1e100]]))
    gpu.forward()
    try:gpu.backward()
    except OverflowError:pass
    else:raise AssertionError('backward overflow accepted')
    for call in [gpu.download_gradients,lambda:gpu.sgd(.001)]:
      try:call()
      except RuntimeError:pass
      else:raise AssertionError('stale gradients usable after failed backward')
    gpu.upload_output_gradient(np.array([[1e-100]]));gpu.backward();near(gpu.download_gradients().layers[0].denominators,[[[-1e200]]],atol=0)
    # Unsafe denominator candidates accepted by SGD, rejected on subsequent execution.
    l=kan.Layer(1,1,config(0,1,0,1));l.set_rational_parameters(np.array([[[1.]]]),np.array([[[0.]]]),np.zeros(1))
    gpu=kan.ResidentNetwork(kan.Network([l]),1);gpu.upload_input(np.array([[1.]]));gpu.upload_output_gradient(np.array([[-1.]]));gpu.forward();gpu.backward();gpu.sgd(1)
    assert gpu.download_parameters().layers[0].denominators[0,0,0]==-1
    try:gpu.forward()
    except ValueError:pass
    else:raise AssertionError('candidate pole was executed')
    print('PASS backward-only overflow invalidation/recovery and unsafe SGD candidate next-execution guard')

def warp_tails():
    if not kan.cuda_available():return
    l=kan.Layer(3,2,config(5,4));l.set_rational_parameters(np.sin(np.arange(36)).reshape(2,3,6)*.03,np.cos(np.arange(24)).reshape(2,3,4)*.007,np.array([.01,-.02]))
    cpu=kan.Network([l]);gpu=kan.ResidentNetwork(cpu,43);alloc=gpu.workspace_allocations
    for batch in [1,31,32,33,37,0,41]:
      x=(np.sin(np.arange(batch*3)*.17)*.6).reshape(batch,3);u=(np.cos(np.arange(batch*2)*.13)*.09).reshape(batch,2)
      gpu.upload_input(x);gpu.upload_output_gradient(u);gpu.forward();gpu.backward(.13)
      near(gpu.download_output(),cpu.forward(x));gg=gpu.download_gradients();cg=cpu.backward(x,u)
      near(gg.input,cg.input)
      for name in ['input','denominators','bias']:near(getattr(gg.layers[0],name),getattr(cg.layers[0],name))
      near(gg.layers[0].coefficients,cg.layers[0].coefficients+.13*l.coefficients)
    assert alloc==gpu.workspace_allocations
    print('PASS warp sample tails1/31/32/33/37/0/41,62-parameter block tail,capacity43 and allocation invariance')

independent_scalar();strict();mixed();failures();warp_tails()
