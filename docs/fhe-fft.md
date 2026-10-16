# The Fourier engine — `fhe/fft`

Every expensive inner loop in frost-VM is a multiplication of polynomials
modulo X^N+1 (see [fhe-bool.md](fhe-bool.md) for where they come from).
Doing that directly costs O(N²) coefficient operations; the Fourier transform
turns it into N/2 pointwise complex multiplications. This package is the
transform, its evaluation-domain representation, and the glue to the integer
coefficients the rest of the scheme uses.

N = 1024 throughout (the polynomial degree; all sizes are powers of two).

## What a polynomial here looks like

A ring element is `p ∈ Z[X]/(X^N + 1)` with coefficients stored as Torus32 —
an `int32` whose value represents the real number `v / 2^32 mod 1` (the
"torus" T = R/Z, discretized at 2^-32). Arithmetic on coefficients is plain
int32 add/sub, which *is* arithmetic mod 2^32, i.e. mod 1 on the torus.
Negation of a polynomial is coefficient-wise int32 negation.

Multiplication is **negacyclic**: after ordinary convolution, terms of degree
≥ N fold back with a sign flip (X^N = −1). That sign is what makes the
transform below possible.

## The LagrangeHalfC representation

Because p has real coefficients and lives mod X^N+1, it is fully determined
by its values at the **odd 2N-th roots of unity**:

    w = exp(i·π/N)          (a primitive 2N-th root)
    values: P(w^1), P(w^3), P(w^5), ..., P(w^(2N−1))    — N/2 complex values

These N/2 complex numbers are the *half-spectrum*; the values at even powers
belong to the conjugate polynomial and carry no extra information. The
representation stores exactly these N/2 values, in **bit-reversed order of
the odd exponent index** (`reva[i] = bitreverse of the i-th odd index mod
2N`), in **SoA layout**: reals in `buf[:N/2]`, imaginaries in `buf[N/2:]`.

Why this pays: **pointwise complex multiplication of half-spectra equals
negacyclic polynomial multiplication.** No wraparound handling, no
correction step — the negacyclic sign is baked into evaluating at odd roots
(X^N · w^r = −w^r for odd r). The external product's inner loop (see
fhe-bool.md) is then just: transform both polynomials, multiply N/2 complex
pairs, inverse-transform.

The test suite asserts this contract directly (mirroring and extending
libtfhe's lagrangehalfc_test.cpp):

- `fft` then `ifft` is the identity on full-range random Torus32 polynomials
  (bijectivity),
- the half-spectrum equals *direct* high-precision evaluation at the odd
  roots (the representation is what we claim it is),
- the Fourier-domain product equals an exact integer negacyclic convolution
  computed independently,
- constant/shift semantics: `SetTorusConstant`, `AddTorusConstant`, `AddTo`,
  and `X^a − 1` multiplication match their closed forms.

## The packed real-input FFT

We never transform N complex values; we transform the 2N real numbers formed
by interleaving (p_0, p_1, p_2, ...) padded to 2N, and read off the odd-root
values. Concretely the pipeline is:

1. **Preprocess**: reorder the 2N real inputs and apply N/2 complex twiddle
   factors, producing N/2 complex values.
2. **Core**: a radix-2 Cooley-Tukey FFT of size N/2 over those complex
   values, in stages of butterflies.
3. **Postprocess**: combine butterfly-paired outputs (the Hermitian
   conjugate pairing of a real-input transform) into the N/2 half-spectrum
   values at odd roots.

The scalar model (`spqlios.go`) implements exactly this, with the butterflies
written over a **4-wide SoA layout** — each stage step processes 4 lanes at
once, which is the same data layout the vector kernels use, so the two
implementations can be compared lane-for-lane.

## The AVX kernels

`kernel_amd64.s` is a Go-assembly port of the hand-written spqlios AVX
kernels: forward FFT, inverse FFT, and the LagrangeHalfC entry/exit glue.
Each ymm register holds 4 doubles, so one instruction does 4 lanes of the
SoA layout — the kernels and the scalar model are the *same algorithm at
width 4*.

Two properties make this safe:

- **Operation order is preserved exactly** between the scalar model and the
  assembly. Floating-point arithmetic is not associative, so bit-level
  agreement requires identical operation order, not just identical
  algebra. The test `TestAVXMatchesScalarModel` asserts the two paths are
  **bit-identical** on random inputs — any future change to either path
  that breaks the port shows up as a test failure, not as silent drift.
- **Runtime dispatch**: an init-time CPUID/XGETBV check (bits 27/28 for
  AVX, XGETBV(0) & 6 == 6 for ymm OS state) selects the assembly path;
  otherwise the scalar model runs. The package builds and tests on any
  amd64, and on arm64 through a scalar-only stub build.

Trig tables are stored interleaved as 4+4 blocks (cos4, sin4 per group of 4
indices) so a kernel loads exactly what a butterfly step needs with no
gather. Table construction uses the same range-reduced accurate sin/cos as
the reference path, and is checked against direct high-precision evaluation.

Measured speedup of the AVX path over the scalar model: 3.8–5.1x (close to
the 4x width limit) on Skylake-era hardware; one external-product inner
loop at N=1024 runs ~15 µs.

## The Processor glue

`processor.go` is the boundary the gate layer actually calls:

- `FFT(p []int32) []float64` — Torus32 polynomial → half-spectrum. The
  int32 values are converted to doubles scaled by 2^-32, transformed, and
  scaled by 2/N so that the inverse path returns values in the original
  integer range.
- `IFFTToTorus(values) []int32` — half-spectrum back to Torus32, with the
  inverse scaling and a **round-to-nearest** conversion.

One deliberate deviation from the C reference: libtfhe truncates
(`(int32)(x)`) at the Fourier→Torus32 conversion; we round to nearest. The
two differ by at most 1 LSB — invisible under FHE noise — but rounding
makes integer round-trips exact, which is what lets the test suite assert
bijectivity and exact-convolution equality without fuzz.

## Single-tap operations

Two small ring operations carry the blind rotation (fhe-bool.md):

- `mulByXai(p, a, source)` — multiply by X^a: a rotation of coefficients
  with sign flips where the rotation crosses the X^N = −1 boundary. Free of
  multiplications; it is an isometry, which is why blind rotation's noise
  grows additively rather than multiplicatively.
- `mulByXaiMinusOne(p, a, source)` — multiply by X^a − 1: one rotation plus
  a negated copy. This is the per-key-bit step of the blind rotation; its
  cost is one pass over the coefficients, no FFT.

## Files

| file | contents |
|---|---|
| `tables.go` | trig table construction (forward/inverse), range-reduced accurate sin/cos |
| `spqlios.go` | scalar model: radix-2 stages over 4-wide SoA, packed real-input transform |
| `kernel_amd64.s` | AVX (ymm) forward/inverse kernels + LagrangeHalfC glue |
| `lagrange.go` | LagrangeHalfC pointwise ops: complex mul, Clear, constants, X^a−1 |
| `processor.go` | Torus32 ↔ evaluation domain, 2/N scaling, round-to-nearest |
| `cpu.go` | CPUID/XGETBV feature detection |
