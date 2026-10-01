# M4 CPU executable RED

2026-10-01. Added independent scalar and layer/network acceptance tests plus
public API and throwing stubs, before numerical implementation.

`scripts/build.ps1 -BuildDirectory build-m4-cpu -Test` built successfully.
`m4_rational` fails all four cases and `m4_layer` fails all five cases with
`M4 RED: rational validation/evaluation/layer pending`. Full output is retained
in `M4-cpu-red.log`. Package tests also fail because the root-owned installed
consumer already exercises the rational stub. Existing numerical tests pass.

The rational constructor uses a constrained forwarding overload requiring exactly
`RationalConfig`; it delegates to a private tagged constructor. This preserves
the established `Layer(inputs,outputs,{})` spelling for default basis layers
while supporting `Layer(inputs,outputs,RationalConfig{})` and Python construction.
Adding two unconstrained aggregate overloads would make the established spelling
ambiguous. Numerical acceptance and defaults are unchanged.

The Visual Studio installer directory was prepended to PATH for this execution
so the developer-environment script could find `vswhere.exe`.
