"""Frozen benchmark label mapping (backlog R5).

R5 renamed the frozen suites and their backend labels by feature. Recorded CSVs
keep the old labels; comparison tooling maps them to the current ones with
`normalize`, so baselines recorded before R5 compare with runs made after it.
Cases, families, columns, options and checksums are unchanged by the rename.
"""

# Suite: executable `<suite>_benchmark`, file prefix in ABBA directories and
# `frozen-<suite>.csv` of scripts/bench_baseline.sh.
SUITES = {
    "m2": "basis",     # Chebyshev, Legendre, Jacobi, Hermite, Fourier, Gaussian RBF
    "m3": "local",     # B-spline, Mexican hat, trainable RBF
    "m4": "rational",  # rational 1/1, 3/2, 6/4
}

# Backend column. cpu_full did not change.
BACKENDS = {
    "m1_host_full": "legacy_full",
    "m2_resident_full": "resident_full",
    "m3_resident_full": "resident_full",
    "m4_resident_full": "resident_full",
    "m2_transfer_full": "transfer_full",
    "m3_transfer_full": "transfer_full",
    "m4_transfer_full": "transfer_full",
}


def suite(name):
    """Current suite name for an old or current one."""
    return SUITES.get(name, name)


def backend(name):
    """Current backend label for an old or current one."""
    return BACKENDS.get(name, name)


def is_suite(name):
    return name in SUITES or name in SUITES.values()
