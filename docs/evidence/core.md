# M1 CPU layer/network TDD evidence

Journeys derived from introduction.md and docs/CONTRACT.md: configure edge expansions,
evaluate batched arbitrary topology, differentiate an externally supplied loss and
train without implicit averaging or parameter mutation during backward.

## RED (2026-09-30)

Before implementation, compiled runnable tests against explicit throwing stubs:

```powershell
g++ -std=c++20 -Iinclude -Itests tests/layer_test.cpp src/layer.cpp -o build-red/layer_test.exe
./build-red/layer_test.exe
g++ -std=c++20 -Iinclude -Itests tests/network_test.cpp src/layer.cpp src/network.cpp -o build-red/network_test.exe
./build-red/network_test.exe
```

Both compilations succeeded. Layer: 0/11 passed, failures `Layer not implemented`.
Network: 0/7 passed, failures `Network not implemented` / `Layer not implemented`.
No production mathematics existed at this gate. GREEN results will be recorded after implementation.

## GREEN (2026-09-30)

Implemented owned parameters, checked array sizes, finite data/results, fixed-order
batched edge contractions, analytic vector-Jacobian products, compatible arbitrary
topology and atomic validated SGD. Built MSVC 19.50.35724 using `./scripts/build.ps1 -Test`.
Initial integrated CTest run: layer/network/example passed; basis 13/14 due to a
test oracle assuming extended `long double` on MSVC (separate agent fixes that oracle).
Direct verification of the core GREEN targets:

```powershell
./build/layer_test.exe       # 11/11 passed
./build/network_test.exe     # 7/7 passed
./build/fit_polynomial.exe   # holdout MSE 2.94787770627e-26
```

| Guarantee | Tests |
|---|---|
| Correct output/index layout and explicit batch sum semantics | known_forward_layout_and_batch, backward_sums_without_averaging_or_mutation |
| All six layer families: input/coefficient/bias gradients | all_basis_layer_gradients_match_finite_differences |
| Multilayer chain-rule and mixed families | network_gradients_match_finite_differences |
| Positive compatible dimensions, zero batch, bad shapes, checked size overflow | dimensions/batches tests in layer_test.cpp and network_test.cpp |
| Invalid/nonfinite data rejected; updates atomic | parameter_set_is_validated_before_mutation, sgd_invalid_update_is_atomic, network_sgd_is_atomic_across_layers |
| Actual training and independent holdout prediction | sgd_training_fits_polynomial, examples/fit_polynomial.cpp |

CPU coverage, memory checks, package consumer and independent review will be recorded
in the final M1 evidence report. No GPU performance claim is made by this stage.
