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
