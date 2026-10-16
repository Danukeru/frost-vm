# Status, verification, costs, roadmap

## Status

All milestones done, each verified at its own level and end-to-end:

1. **`fhe/fft` — the Fourier engine** (done): scalar model + hand-written
   AVX kernels, bit-identical by test; the representation is verified
   against exact integer convolutions and direct evaluation at the odd
   roots.
2. **`fhe/bool` — the TFHE gate layer** (done, three-round incremental
   oracle review): full gate truth tables decrypt correctly through the
   real bootstrap; blind-rotation and external-product invariants checked
   with exact integer arithmetic; multi-bootstrap chains survive noise
   accumulation.
3. **`vm/` — the oblivious RV64I machine** (done): full base-ISA coverage
   (42/42, see vm-machine.md), word/ALU tables against Go's own RV64
   semantics, three hand-encoded oblivious programs, and real
   clang-compiled C programs running end-to-end.
4. **`programs/` + ELF loader** (done): the fibonacci demo (first 3
   values) runs obliviously for 10 cycles and data memory decrypts to
   0, 1, 1; the machine halts itself via ECALL. An earlier 20-value
   version (134 cycles, 7.4 h) validated the loop/branch path — the
   program-counter trace matched the compiled loop exactly — and exposed
   the one real machine-layer bug found so far (load-extension width).
   Its wrong-value streak was diagnosed *from the decrypted values alone*
   (they matched a single broken-path hypothesis arithmetically, 5/5)
   before any re-run; the bug is now pinned by a decisive regression
   test.

## Verification strategy

- **Every layer is table-tested against independent semantics**: the
  Fourier core against exact integer convolutions and direct evaluation;
  the gate layer against full truth tables through the real bootstrap and
  against exact algebraic invariants; the machine's words and ALU against
  Go's own RV64 arithmetic; the loader against exact clang encodings.
- **The scalar and AVX Fourier paths are bit-identical** by construction
  and by test — no silent drift between them is possible.
- **Deterministic seeds**: a Context's keys and noise are seeded; test
  values are reproducible run to run.
- **Decisive pins over volume**: each fixed bug gets one test that fails
  exactly for its signature (load-extension width, immediate bit
  pairing, TGSW diagonal convention, phaseToT32 domain), so regressions
  name themselves.

## Reproducibility

`docker build --pull -t frost-vm .` runs
`go vet && go build && go test ./...` inside
`golang:1.24-bookworm@sha256:1a6d4452...` — the whole suite (~50 min of
real TFHE) in a byte-reproducible image. The multi-hour end-to-end runs
are env-gated (`FROST_RUN_FIB=1`) and not part of the image build.

## Cost model, honestly

At test parameters: ~3.5 ms per AND/OR/XOR gate, ~16 ms per MUX. One
oblivious cycle at (32 instructions, 16 registers, 128 bytes RAM) is ~5k
gates ≈ 200 s, dominated by the RAM/register masked-OR decodes and the
three 64-bit barrel shifters. The machine test suite (~17 cycles) runs
~11 min; the load-extension pin ~40 min; the 10-cycle fibonacci demo
~33 min; a 134-cycle loop program 7.4 h. This is the honest price of not
looking: per-cycle work is fixed by the machine's size, independent of
the program's control flow.

## Roadmap

1. **CLI**: `run program.elf` → decrypted register/memory state +
   cycle/gate statistics.
2. **Performance**: the big algorithmic lever is multi-bit / programmable
   bootstrapping with packed test vectors (one bootstrap evaluates
   several bits or an arbitrary function); the easy engineering lever is
   parallelizing the external-product terms across cores (they are
   independent until the final accumulation).
3. **ISA depth**: RV64M via shift-add (the ALU already has the adders
   and shifters it needs), larger machine geometries.
