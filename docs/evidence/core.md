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
