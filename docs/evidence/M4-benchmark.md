# M4 frozen matched rational benchmark protocol

Frozen 2026-10-01 before resident implementation or tuning. The original
`benchmarks/m4_benchmark.cpp` defines degree pairs 1/1, 3/2, 6/4, center 0.1,
scale 1.2, epsilon 1e-8, topologies 16 -> 24 -> 8 and 64 -> 64 -> 32 -> 16,
batch 32/1024 and double precision. It fixes deterministic sine numerators,
cosine denominators/bias, sine input and upstream and SGD rate 0.001.

Each executor starts from identical parameters and performs two warmups then
seven measured evolving forward/backward/SGD steps. CPU backward recomputes
activations. Resident complete calls include scalar status validation and stream
synchronization. Transfer complete calls additionally upload inputs/upstream and
download output and every gradient before SGD. Setup is separate. Complete-call
timing uses host steady_clock; no kernel-only speedup is accepted.

Every output, input VJP, numerator/denominator/bias VJP and final parameter is
verified elementwise against CPU at 2e-10 absolute-plus-relative tolerance.
Steady resident verification uses an untimed identical trajectory replay whose
parameters must exactly match the measured executor. Allocation counts must
remain two. CSV retains every sample, interpolated median/IQR and checksums.
Fixture changes require explicit evidence; tuning uses balanced baseline/final/
final/baseline comparisons of the same frozen case. Profiling durations are
diagnostic only. Hardware/profiler constraints and raw artifacts follow below.
