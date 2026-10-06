"""Backlog C9: PyTorch counterpart of train_bench.cpp (matched and realistic protocols).

The model is torch_reference.py's: per layer h = cheb(h, K).flatten(1) @ C.T + b,
Chebyshev K=7, manual SGD. Protocols:
  matched    x and the upstream stay resident; step = forward + backward(up) + SGD
             (exactly torch_reference.py).
  realistic  every step copies a new (input, target) batch from a pinned host
             dataset (4 batches, cycled) with non_blocking=True, then
             forward + mse_loss + backward + SGD. The loss is never read back.
Modes:
  eager      plain eager PyTorch.
  graph      the whole step captured once with torch.cuda.graph and replayed
             (static input/target buffers; realistic copies into them first).
  compile-<backend>
             torch.compile(forward, backend=<backend>) with the eager SGD:
             'inductor' (needs Triton) or 'cudagraphs' (AOTAutograd forward and
             backward replayed as CUDA graphs, no Triton). A backend that fails
             is reported on stderr and the script exits with status 3.
Usage: torch_bench.py <f64|f32|tf32> <matched|realistic> <eager|graph|compile-BACKEND> [windows] [case]
"""
import sys
import time

import torch
import torch.nn.functional as F

CASES = [
    ("64x64x32x16-b1024", (64, 64, 32, 16), 1024, 200),
    ("16x24x8-b1024", (16, 24, 8), 1024, 200),
    ("256x256x256x10-b8192", (256, 256, 256, 10), 8192, None),
    ("1024x1024x1024-b4096", (1024, 1024, 1024), 4096, None),
]
K = 7
RATE = 1e-3


def cheb(x, k):
    t = [torch.ones_like(x), x]
    for _ in range(k - 2):
        t.append(2 * x * t[-1] - t[-2])
    return torch.stack(t, -1)


def forward(params, x):
    h = x
    for c, b in params:
        h = cheb(h, K).flatten(1) @ c.T + b
    return h


def sgd(params):
    with torch.no_grad():
        for c, b in params:
            c -= RATE * c.grad
            b -= RATE * b.grad
            c.grad = None
            b.grad = None


def wave(count, scale, frequency, phase, dtype):
    i = torch.arange(count, dtype=torch.float64)
    return (scale * torch.sin(frequency * i + phase)).to(dtype)


def main():
    precision, protocol, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    windows = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    only = sys.argv[5] if len(sys.argv) > 5 else ""
    dtype = torch.float64 if precision == "f64" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    print("impl,precision,protocol,mode,interval,case,window,steps,ms_per_step,checksum")
    for name, dims, batch, steps in CASES:
        if only and only != name:
            continue
        if steps is None:
            fp64 = dtype == torch.float64
            steps = (10 if fp64 else 50) if dims[0] == 256 else (2 if fp64 else 20)
        torch.manual_seed(0)
        params = [(torch.randn(o, i * K, device="cuda", dtype=dtype).mul_(0.1 / (i * K) ** 0.5).requires_grad_(),
                   torch.zeros(o, device="cuda", dtype=dtype).requires_grad_()) for i, o in zip(dims, dims[1:])]
        xs = [wave(batch * dims[0], 0.95, 0.113 + 0.017 * b, 0.3 * b, dtype).view(batch, dims[0]).pin_memory() for b in range(4)]
        ts = [wave(batch * dims[-1], 0.1, 0.071 + 0.013 * b, 0.7 * b, dtype).view(batch, dims[-1]).pin_memory() for b in range(4)]
        up = (wave(batch * dims[-1], 1.0 / batch, 0.07, 0.1, dtype).view(batch, dims[-1])).cuda()
        x0 = xs[0].cuda()
        fwd = forward
        if mode.startswith("compile-"):
            try:
                fwd = torch.compile(forward, backend=mode.split("-", 1)[1])
            except Exception as error:  # pragma: no cover - reported
                print(f"# compile unavailable: {error!r}", file=sys.stderr)
                return 3

        if protocol == "matched":
            def body(x=x0):
                fwd(params, x).backward(up)
                sgd(params)
        else:
            static_x = torch.empty(batch, dims[0], device="cuda", dtype=dtype)
            static_t = torch.empty(batch, dims[-1], device="cuda", dtype=dtype)

            def body(x=None, t=None):
                x = static_x if x is None else x
                t = static_t if t is None else t
                F.mse_loss(fwd(params, x), t).backward()
                sgd(params)

        counter = [0]
        graph = None
        if mode == "graph":
            if protocol == "realistic":
                static_x.copy_(xs[0])
                static_t.copy_(ts[0])
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    body()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body()

        def step():
            k = counter[0] % 4
            counter[0] += 1
            if graph is not None:
                if protocol == "realistic":
                    static_x.copy_(xs[k], non_blocking=True)
                    static_t.copy_(ts[k], non_blocking=True)
                graph.replay()
            elif protocol == "matched":
                body()
            else:
                body(xs[k].to("cuda", non_blocking=True), ts[k].to("cuda", non_blocking=True))

        try:
            for _ in range(max(3, steps // 10)):
                step()
            torch.cuda.synchronize()
        except Exception as error:  # compile backends failing at first call
            print(f"# {mode} unavailable for {name}: {type(error).__name__}: {str(error).splitlines()[0][:200]}", file=sys.stderr)
            return 3
        for w in range(windows):
            start = time.perf_counter()
            for _ in range(steps):
                step()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) / steps * 1e3
            checksum = sum(float(c.detach().sum()) for c, _ in params) if w + 1 == windows else 0.0
            print(f"torch,{precision},{protocol},{mode},0,{name},{w},{steps},{ms:.4f},{checksum:.10g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
