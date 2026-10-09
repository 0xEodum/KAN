## f32-256x256x256x10-b8192-branch0 (55 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| ampere_sgemm_128x64_tn | 2.00 | 824.82 |
| cutlass::Kernel2 | 2.00 | 663.44 |
| ampere_sgemm_128x64_nn | 1.00 | 364.18 |
| backward_finish_kernel | 3.00 | 262.67 |
| basis_kernel | 3.00 | 252.31 |
| ampere_sgemm_32x128_tn | 1.00 | 95.16 |
| ampere_sgemm_128x32_nt | 1.00 | 75.59 |
| ampere_sgemm_128x32_nn | 1.00 | 71.70 |
| bias_kernel | 3.00 | 45.37 |
| gemvNSP_kernel | 1.00 | 28.51 |
| std::enable_if | 2.00 | 25.49 |
| candidate_kernel | 1.00 | 11.62 |
| cublasLt::splitKreduce_kernel | 3.00 | 6.55 |
| commit_kernel | 1.00 | 3.94 |
| fill_kernel | 0.02 | 0.03 |
| **total** | 25.02 | 2731.38 |

## f32-256x256x256x10-b8192-branch1 (55 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| ampere_sgemm_128x64_tn | 4.00 | 988.00 |
| cutlass::Kernel2 | 5.00 | 939.85 |
| ampere_sgemm_128x64_nn | 1.00 | 364.74 |
| backward_finish_kernel | 3.00 | 263.69 |
| basis_kernel | 3.00 | 249.91 |
| ampere_sgemm_32x128_tn | 2.00 | 115.40 |
| residual_finish_kernel | 3.00 | 83.53 |
| ampere_sgemm_128x32_nn | 1.00 | 79.80 |
| ampere_sgemm_128x32_nt | 1.00 | 76.36 |
| silu_kernel | 3.00 | 63.57 |
| bias_kernel | 3.00 | 45.48 |
| check_kernel | 3.00 | 31.38 |
| gemvNSP_kernel | 1.00 | 28.59 |
| std::enable_if | 2.00 | 25.61 |
| ampere_sgemm_32x32_sliced1x4_nt | 1.00 | 18.32 |
| ampere_sgemm_32x128_nn | 1.00 | 13.22 |
| candidate_kernel | 1.00 | 12.86 |
| cublasLt::splitKreduce_kernel | 4.00 | 8.91 |
| commit_kernel | 1.00 | 3.93 |
| fill_kernel | 0.02 | 0.03 |
| **total** | 43.02 | 3413.18 |

## f32-64x64x32x16-b1024-branch0 (440 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| forward_dot_kernel | 2.00 | 38.53 |
| cutlass::Kernel2 | 2.00 | 14.46 |
| backward_finish_kernel | 3.00 | 12.70 |
| basis_kernel | 3.00 | 10.43 |
| ampere_sgemm_32x32_sliced1x4_tn | 1.00 | 9.90 |
| ampere_sgemm_32x128_nn | 2.00 | 9.43 |
| gemvNSP_kernel | 2.00 | 9.15 |
| cublasLt::splitKreduce_kernel | 3.00 | 7.23 |
| parameter_partial_kernel | 1.00 | 7.19 |
| commit_kernel | 1.00 | 3.47 |
| bias_kernel | 1.00 | 1.74 |
| candidate_kernel | 1.00 | 1.72 |
| fill_kernel | 0.00 | 0.00 |
| **total** | 22.00 | 125.97 |

## f32-64x64x32x16-b1024-branch1 (440 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| forward_dot_kernel | 2.00 | 38.79 |
| residual_forward_kernel | 3.00 | 38.30 |
| parameter_partial_kernel | 3.00 | 15.68 |
| cutlass::Kernel2 | 2.00 | 14.41 |
| backward_finish_kernel | 3.00 | 12.86 |
| basis_kernel | 3.00 | 10.47 |
| ampere_sgemm_32x32_sliced1x4_tn | 1.00 | 9.94 |
| residual_finish_kernel | 3.00 | 9.75 |
| ampere_sgemm_32x128_nn | 2.00 | 9.47 |
| gemvNSP_kernel | 2.00 | 9.13 |
| cublasLt::splitKreduce_kernel | 4.00 | 9.02 |
| ampere_sgemm_32x32_sliced1x4_nt | 1.00 | 4.78 |
| commit_kernel | 1.00 | 3.46 |
| bias_kernel | 1.00 | 1.76 |
| candidate_kernel | 1.00 | 1.69 |
| fill_kernel | 0.00 | 0.00 |
| **total** | 32.00 | 189.51 |

## f64-256x256x256x10-b8192-branch0 (13 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| cutlass::Kernel2 | 6.00 | 72905.75 |
| basis_kernel | 3.00 | 5061.07 |
| parameter_partial_kernel | 1.00 | 1116.45 |
| backward_finish_kernel | 3.00 | 821.27 |
| ampere_dgemm_64x64_nn | 1.00 | 817.81 |
| bias_kernel | 3.00 | 84.83 |
| gemvNSP_kernel | 2.00 | 51.23 |
| candidate_kernel | 1.00 | 22.56 |
| cublasLt::splitKreduce_kernel | 2.00 | 5.39 |
| commit_kernel | 1.00 | 3.50 |
| fill_kernel | 0.08 | 0.13 |
| **total** | 23.08 | 80889.99 |

## f64-256x256x256x10-b8192-branch1 (13 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| cutlass::Kernel2 | 12.00 | 82936.55 |
| basis_kernel | 3.00 | 5078.78 |
| parameter_partial_kernel | 2.00 | 1326.31 |
| ampere_dgemm_64x64_nn | 2.00 | 992.30 |
| backward_finish_kernel | 3.00 | 824.61 |
| silu_kernel | 3.00 | 657.40 |
| residual_finish_kernel | 3.00 | 574.66 |
| bias_kernel | 3.00 | 84.40 |
| gemvNSP_kernel | 2.00 | 51.07 |
| check_kernel | 3.00 | 50.06 |
| candidate_kernel | 1.00 | 28.87 |
| cublasLt::splitKreduce_kernel | 2.00 | 5.42 |
| commit_kernel | 1.00 | 3.28 |
| fill_kernel | 0.08 | 0.12 |
| **total** | 40.08 | 92613.83 |

## f64-64x64x32x16-b1024-branch0 (220 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| parameter_partial_kernel | 3.00 | 408.28 |
| cutlass::Kernel2 | 3.00 | 245.77 |
| basis_kernel | 3.00 | 95.69 |
| dgemm_largek | 1.00 | 65.26 |
| forward_dot_kernel | 1.00 | 44.57 |
| backward_finish_kernel | 3.00 | 25.96 |
| commit_kernel | 1.00 | 7.63 |
| bias_kernel | 2.00 | 4.07 |
| candidate_kernel | 1.00 | 1.94 |
| scal_kernel | 1.00 | 1.11 |
| fill_kernel | 0.00 | 0.01 |
| **total** | 19.00 | 900.28 |

## f64-64x64x32x16-b1024-branch1 (220 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| parameter_partial_kernel | 6.00 | 486.47 |
| cutlass::Kernel2 | 3.00 | 246.17 |
| residual_forward_kernel | 3.00 | 235.51 |
| basis_kernel | 3.00 | 96.49 |
| dgemm_largek | 1.00 | 71.96 |
| residual_finish_kernel | 3.00 | 55.93 |
| forward_dot_kernel | 1.00 | 44.63 |
| backward_finish_kernel | 3.00 | 28.14 |
| bias_kernel | 2.00 | 4.10 |
| commit_kernel | 1.00 | 3.44 |
| candidate_kernel | 1.00 | 1.98 |
| scal_kernel | 1.00 | 1.13 |
| fill_kernel | 0.00 | 0.01 |
| **total** | 28.00 | 1275.95 |

