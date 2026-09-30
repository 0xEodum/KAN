# M1 basis TDD evidence

The six basis families and derivatives are implemented locally from mathematical
definitions, with no imported KAN source. Polynomial recurrence conventions were
checked against [NIST DLMF 18.9](https://dlmf.nist.gov/18.9), including Jacobi's
continuous initial coefficients for alpha + beta = -1 and 0 and physicists' Hermite.

## RED

Command (GCC/UCRT64):

```powershell
g++ -std=c++20 -Wall -Wextra -Wpedantic -Iinclude -Itests tests/basis_test.cpp src/basis.cpp -o build-basis/basis-tests.exe
./build-basis/basis-tests.exe
```

The initial stub threw `std::logic_error("basis not implemented")`.
Compilation succeeded; all 11 initial test cases failed; test process exit code 1.
After clarifying extreme Gaussian behavior, the suite contains 12 RED cases.
The finite Gaussian result must remain computable across subtraction/division
overflow and genuine underflow tails; an overflowing returned derivative still fails.

Tests use fourth-degree closed forms, endpoint identities, Jacobi's independent
finite binomial sum and derivative identity, Fourier ordering, Gaussian direct
formula checks, and central differences at interior/outside-domain points.
Validation covers unknown enums, impossible/zero sizes, invalid selected-family
parameters, irrelevant parameters, nonfinite inputs, and numeric overflow.

## GREEN

The same standalone compile/run succeeded with `-Werror` added: 14/14 passed,
compiler exit code 0, test exit code 0. Recurrences analytically propagate the
derivative without finite differences in production. Jacobi coefficients use
half-sums and ratios, tested with alpha = beta = 1e200 at x = 0.

An additional Gaussian regression was first executed against the initial GREEN
implementation: 13/14 passed, exit code 1, because a derivative became zero when
the Gaussian value underflowed. For width = denorm_min and normalized distance 30,
an independent long-double oracle demonstrated a representable derivative. The
fix evaluates subnormal/underflow derivative magnitudes in logarithmic form; the
same 14-case suite then passed. Opposite-sign extreme centers/inputs use scaled
subtraction so a wide Gaussian retains its finite value. True distant tails return
zero and derivatives that exceed double range still raise overflow_error.

All builds and executables remain in ignored `build-basis/`. The deterministic
suite includes six-family central differences and independent polynomial oracles;
integration and independent review are coordinated by the main project lane.

## MSVC portability follow-up

The integrated MSVC run exposed an oracle portability issue: MSVC's `long double`
has the same range as `double`, so the test's independent `exp(-900)` oracle
underflowed before division and failed with 13/14 passing. Production evaluation
was unaffected. The regression now uses a precomputed constant from Python
`decimal.Decimal` with precision 100, evaluating `-60 * exp(-900) / (2 ** -1074)`:

```text
-1.657039574215751920267356912558436470529179262842470867148420005915681625311244551783798700124454242E-66
```

After that test-only change, standalone GCC with `-Werror` passed 14/14. MSVC
14.50.35717 Release, after initializing `vcvars64.bat`, also passed 14/14 using
`cmake --build build --target basis_test` followed by `build/basis_test.exe`.
Both builds and test runs exited 0. Calling the MSVC build in a plain shell without
the compiler environment had failed to locate `cstddef`; rerunning in the initialized
environment resolved that setup issue.

## Independent review: Jacobi endpoint cancellation RED

Independent review found that alpha = 1e17, beta = 0, size = 5, x = -1
returned `[1, 0, -0.5, 0, 0.375]` instead of `[1, -1, 1, -1, 1]`.
The general recurrence lost the small endpoint value when subtracting large
terms; degree-2 derivative also lost its approximately -1e17 value.

Two new test cases use [NIST DLMF 18.6.T1](https://dlmf.nist.gov/18.6.T1)
endpoint rising-factorial identities and [18.9.E15](https://dlmf.nist.gov/18.9.E15)
for derivatives. They exercise both endpoints, asymmetric parameters up to 1e70,
the maximum finite parameter with a finite first derivative, and true endpoint
value/derivative overflow. The standalone GCC command above with `-Werror`
compiled successfully and returned **14/16 passed**, test exit code 1. Both new
cases failed against the pre-fix implementation; all previous cases still passed.

## Independent review: Jacobi endpoint cancellation GREEN

For exactly x = -1 or +1, Jacobi evaluation now propagates endpoint values
through their rising-factorial identities and computes derivatives through the
shifted-parameter endpoint identity. The derivative prefactor uses half-sums to
preserve representable results for maximum finite parameters. No coordinate
clipping or near-endpoint substitution is introduced; other coordinates retain
the general recurrence. Finite-result guards remain on endpoint values and
derivatives, so true range overflow raises `std::overflow_error`.

The same standalone GCC compile with `-Werror` and test run passed **16/16**,
compiler and test exit codes 0. Only the basis implementation, tests, and this
evidence were changed; the shared integrated build was left to the main lane.

## Final review: Jacobi parameters near -1 RED

The final independent review reproduced alpha = nextafter(-1, 0),
beta = nextafter(alpha, 0), size = 3, x = 0 producing P2 = -0.14062500000000003,
against the finite-sum reference -0.25000000000000006. Expressions `n + half_sum`
and `(n+1)/2 + half_sum` lost relative precision in the first recurrence step when
both parameters approached -1. The first derivative slope had the same issue.

A new regression tests both neighboring doubles above -1 in all four alpha/beta
combinations, seven coordinates from -1 to +1, and degrees zero through seven.
Values use the existing independent finite binomial sum, derivatives use the
shifted-polynomial identity, and the first slope is additionally compared with
relative scaling. The standalone GCC warning-as-error build succeeded, while
the test run returned **16/17 passed**, exit code 1, before implementation changes.
