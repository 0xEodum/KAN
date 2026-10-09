## f32-256x256x256x10-b8192-branch0 (55 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| ampere_sgemm_128x64_tn | 2.00 | 794.96 |
| cutlass::Kernel2 | 2.00 | 629.58 |
| ampere_sgemm_128x64_nn | 1.00 | 350.13 |
| backward_finish_kernel | 3.00 | 255.21 |
| basis_kernel | 3.00 | 249.72 |
| ampere_sgemm_32x128_tn | 1.00 | 91.47 |
| ampere_sgemm_128x32_nt | 1.00 | 82.26 |
| ampere_sgemm_128x32_nn | 1.00 | 70.28 |
| bias_kernel | 3.00 | 44.66 |
| gemvNSP_kernel | 1.00 | 27.21 |
| std::enable_if | 2.00 | 25.52 |
| candidate_kernel | 1.00 | 11.50 |
| cublasLt::splitKreduce_kernel | 3.00 | 6.32 |
| commit_kernel | 1.00 | 3.73 |
| fill_kernel | 0.02 | 0.03 |
| **total** | 25.02 | 2642.58 |

## f32-256x256x256x10-b8192-branch1 (55 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| ampere_sgemm_128x64_tn | 4.00 | 946.99 |
| cutlass::Kernel2 | 5.00 | 805.39 |
| ampere_sgemm_128x64_nn | 1.00 | 350.30 |
| basis_kernel | 3.00 | 319.65 |
| backward_finish_kernel | 3.00 | 263.98 |
| ampere_sgemm_32x128_tn | 2.00 | 112.29 |
| residual_finish_kernel | 3.00 | 82.52 |
| ampere_sgemm_128x32_nt | 1.00 | 73.73 |
| ampere_sgemm_128x32_nn | 1.00 | 70.31 |
| bias_kernel | 3.00 | 44.47 |
| gemvNSP_kernel | 1.00 | 27.41 |
| std::enable_if | 2.00 | 25.21 |
| ampere_sgemm_32x32_sliced1x4_nt | 1.00 | 17.36 |
| cublasLt::splitKreduce_kernel | 6.00 | 16.44 |
| candidate_kernel | 1.00 | 15.76 |
| ampere_sgemm_32x128_nn | 1.00 | 12.62 |
| commit_kernel | 1.00 | 3.70 |
| fill_kernel | 0.02 | 0.03 |
| **total** | 39.02 | 3188.17 |

## f32-64x64x32x16-b1024-branch0 (440 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| forward_dot_kernel | 2.00 | 38.57 |
| cutlass::Kernel2 | 2.00 | 14.50 |
| backward_finish_kernel | 3.00 | 12.90 |
| basis_kernel | 3.00 | 10.43 |
| ampere_sgemm_32x32_sliced1x4_tn | 1.00 | 9.95 |
| ampere_sgemm_32x128_nn | 2.00 | 9.43 |
| gemvNSP_kernel | 2.00 | 9.29 |
| parameter_partial_kernel | 1.00 | 7.23 |
| cublasLt::splitKreduce_kernel | 3.00 | 7.07 |
| commit_kernel | 1.00 | 3.53 |
| bias_kernel | 1.00 | 1.74 |
| candidate_kernel | 1.00 | 1.70 |
| fill_kernel | 0.00 | 0.00 |
| **total** | 22.00 | 126.35 |

## f32-64x64x32x16-b1024-branch1 (440 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| forward_dot_kernel | 2.00 | 45.95 |
| parameter_partial_kernel | 3.00 | 15.81 |
| cutlass::Kernel2 | 2.00 | 14.45 |
| backward_finish_kernel | 3.00 | 12.89 |
| basis_kernel | 3.00 | 11.27 |
| ampere_sgemm_32x32_sliced1x4_tn | 1.00 | 10.11 |
| residual_dot_kernel | 1.00 | 9.76 |
| ampere_sgemm_32x128_nn | 2.00 | 9.45 |
| residual_finish_kernel | 3.00 | 9.36 |
| cublasLt::splitKreduce_kernel | 4.00 | 9.35 |
| gemvNSP_kernel | 2.00 | 9.12 |
| ampere_sgemm_32x32_sliced1x4_nt | 1.00 | 4.71 |
| commit_kernel | 1.00 | 3.55 |
| bias_kernel | 1.00 | 1.74 |
| candidate_kernel | 1.00 | 1.66 |
| fill_kernel | 0.00 | 0.00 |
| **total** | 30.00 | 169.18 |

## f64-256x256x256x10-b8192-branch0 (13 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| cutlass::Kernel2 | 6.00 | 76277.55 |
| basis_kernel | 3.00 | 4990.10 |
| parameter_partial_kernel | 1.00 | 1172.91 |
| ampere_dgemm_64x64_nn | 1.00 | 860.40 |
| backward_finish_kernel | 3.00 | 833.01 |
| bias_kernel | 3.00 | 84.05 |
| gemvNSP_kernel | 2.00 | 51.52 |
| candidate_kernel | 1.00 | 22.42 |
| cublasLt::splitKreduce_kernel | 2.00 | 5.70 |
| commit_kernel | 1.00 | 3.57 |
| fill_kernel | 0.08 | 0.13 |
| **total** | 23.08 | 84301.35 |

## f64-256x256x256x10-b8192-branch1 (13 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| cutlass::Kernel2 | 12.00 | 87005.76 |
| basis_kernel | 3.00 | 5435.98 |
| parameter_partial_kernel | 2.00 | 1356.27 |
| ampere_dgemm_64x64_nn | 2.00 | 1010.72 |
| backward_finish_kernel | 3.00 | 824.49 |
| residual_finish_kernel | 3.00 | 162.17 |
| bias_kernel | 3.00 | 85.64 |
| gemvNSP_kernel | 2.00 | 51.65 |
| candidate_kernel | 1.00 | 28.87 |
| cublasLt::splitKreduce_kernel | 2.00 | 5.73 |
| commit_kernel | 1.00 | 3.56 |
| fill_kernel | 0.08 | 0.13 |
| **total** | 34.08 | 95970.98 |

## f64-64x64x32x16-b1024-branch0 (220 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| parameter_partial_kernel | 3.00 | 410.33 |
| cutlass::Kernel2 | 3.00 | 254.83 |
| basis_kernel | 3.00 | 96.16 |
| dgemm_largek | 1.00 | 67.28 |
| forward_dot_kernel | 1.00 | 45.98 |
| backward_finish_kernel | 3.00 | 26.49 |
| bias_kernel | 2.00 | 4.27 |
| commit_kernel | 1.00 | 3.60 |
| candidate_kernel | 1.00 | 1.89 |
| scal_kernel | 1.00 | 1.14 |
| fill_kernel | 0.00 | 0.01 |
| **total** | 19.00 | 911.98 |

## f64-64x64x32x16-b1024-branch1 (220 steps)
| kernel | launches/step | us/step |
|---|---:|---:|
| parameter_partial_kernel | 6.00 | 462.68 |
| cutlass::Kernel2 | 3.00 | 236.05 |
| basis_kernel | 3.00 | 118.19 |
| residual_dot_kernel | 2.00 | 70.27 |
| dgemm_largek | 1.00 | 62.66 |
| forward_dot_kernel | 1.00 | 49.01 |
| residual_finish_kernel | 3.00 | 37.15 |
| backward_finish_kernel | 3.00 | 25.72 |
| bias_kernel | 2.00 | 3.85 |
| commit_kernel | 1.00 | 3.33 |
| candidate_kernel | 1.00 | 1.97 |
| scal_kernel | 1.00 | 1.07 |
| fill_kernel | 0.00 | 0.01 |
| **total** | 27.00 | 1071.97 |

