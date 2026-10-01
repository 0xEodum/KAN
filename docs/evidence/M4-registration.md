# M4 build and acceptance registration

2026-10-01. CMake registers the dedicated rational CPU source, two CPU suites,
resident GPU suite, Python suite and frozen M4 benchmark. CPU test targets build
successfully with the explicit RED stubs at `7379dc0`: scalar 0/4 and layer 0/5,
as independently replayed by the coordinator. These are intentionally failing
tests until the implementation steps pass. Installed-consumer tests will also
exercise the new rational API after integration.

The existing default basis aggregate `Layer(1,1,{})` remains source compatible:
the rational constructor is constrained to a typed RationalConfig, avoiding
ambiguous empty-brace overload resolution. This changes neither the mathematical
contract nor accepted constructor usage from M4-plan.md.
