## Сборка
`C:\Program Files\Microsoft Visual Studio\18\Professional` - building tools:
`C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat`
`C:\Program Files\Microsoft Visual Studio\18\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`
`.venv` - Python 3.13, PyTroch 2.14

## Git
Source of ideas: https://github.com/mintisan/awesome-kan
Upload to the repository: https://github.com/0xEodum/KAN
One example: https://github.com/wtroy2/Quantum-KAN, but it has a number of issues:
- Converting the AST to strings and hashing via __str__ ()
- Parsing strings during numerical computations
- Undefined behaviour (UB) with memory in Eigen
- Hard-coded topology
- A huge amount of duplicate code (copy-paste)
- Namespace pollution and monolithic design
Mathematical and quantum problems (QUBO/KAN)
Rosenberg’s quadratisation is naive; is a symbol engine (SymEngine) even needed here?

[!] None of these projects can be ported into our library as they stand; we must not attempt to adapt interfaces and so on. We are not assembling a project from parts, but building our own.

## Target project:
Modularity, categorisation into KAN types.
For example:
- Orthogonal polynomials (Chebyshev, Legendre, Jacobi, Hermite)
- Harmonic / Wave (Fourier, Wavelets)
- Radial and rational (RBF, Padé)
- Quantum carriers (PQC / Parametrised schemes, Fock modes – experimental section)
- Possibly others, if you know of any further variants.

## Development:
Main stack: C++/CUDA; Python only for short tests/benchmarks, wrappers and very specific libraries. This is necessary as KANs will require non-standard kernels for optimisation.



