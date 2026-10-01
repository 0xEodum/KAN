# M3 frozen matched benchmark protocol

Frozen before CUDA implementation/tuning on 2026-10-01. `m3_benchmark.cpp`
fixes three families (cubic clamped B-spline, normalized MexicanHat, trainable
Gaussian RBF), seven terms, topologies 16 -> 24 -> 8 and 64 -> 64 -> 32 -> 16,
batch 32/1024, double arithmetic, deterministic nonzero parameters, sine inputs
and upstream, and learning rate 0.001. Explicit knots, scales and log widths are
in source. Input/upstream generation occurs outside measurement.

Each backend starts identically, performs two warmups then seven measured
forward/backward/SGD steps on the evolving parameters. CPU backward recomputes
activations; resident backward reuses the preceding forward state. Resident
full calls include scalar numerical validation/synchronization and exclude full
tensor traffic. Transfer full calls additionally upload input/upstream and
download output/all gradients before SGD each step. Setup and final parameter
snapshots are measured separately/outside steady timing. Timings use host
steady_clock and synchronized complete public calls, never kernel-only ratios.

All outputs, input/coefficient/bias/center/log-width VJPs and learned parameter
snapshots are compared elementwise to CPU with 2e-10 absolute-plus-relative
tolerance. Resident verification uses an untimed identical trajectory replay
whose final parameters must equal the timed trajectory. Device allocation count
must stay constant. Each CSV retains all samples, checksums, median and IQR.

Baseline/profiling/tuning results are appended only after real execution.
