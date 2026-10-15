| precision | path | topology | batch | degree | terms | sparse layers | arena MB base -> new | step ms base | step ms new | new/base | forward new/base | checksum rel. diff |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| f32 | dense | 64x64 | 1024 | 5 | 130 | 0 | 109.4 -> 78.9 | 2.974 | 1.095 | 0.368 | 0.334 | 0.0e+00 |
| tf32 | dense | 256x256x256x10 | 8192 | 5 | 130 | 0 | 7572.1 -> 4484.4 | 209.889 | 110.819 | 0.528 | 0.554 | 1.5e-01 |
| f32 | dense | 256x256x256x10 | 8192 | 3 | 130 | 0 | 7572.1 -> 4484.4 | 235.439 | 128.919 | 0.548 | 0.586 | 1.5e-05 |
