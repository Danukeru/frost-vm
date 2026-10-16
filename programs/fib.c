// The end-to-end demo program: ordinary C that compiles to a short RV64I
// instruction stream, runs entirely on encrypted state, and writes its
// result into encrypted data memory where the test decrypts it. It
// exercises the full machine path — LUI/SW prologue, LW/LW/ADD/SW body,
// ECALL self-halt — with no library, no OS, and no way for the compiler
// to fold the computation away. The machine never learns that this
// program is computing fibonacci; it pays the same fixed per-cycle cost
// it pays for any program.
//
// Original file header follows.
//
// fib.c — freestanding RV64I program for the frost-vm oblivious FHE VM.
//
// Computes the first 3 Fibonacci numbers into the `fib` array in data
// memory, then halts with ECALL (the VM's halt instruction). No libc, no
// OS: the VM executes the raw RV64I instruction stream.
//
// Build (see build.sh): clang for riscv64, RV64I base ISA only (the VM
// implements the base integer ISA), medlow absolute addressing, no
// relaxation, custom linker script placing .text at 0 and .data at 0x1000
// (the VM maps data addresses modulo its memory size).
//
// `fib` is volatile on purpose: it forces the real load/add/store
// computation through memory instead of letting clang constant-fold the
// whole sequence away. The VM executes every LW/ADD/SW obliviously.

typedef unsigned int u32;

volatile u32 fib[20];

static inline void halt(void) {
    __asm__ volatile("ecall");
}

void _start(void) {
    fib[0] = 0;
    fib[1] = 1;
    for (unsigned i = 2; i < 3; i++) {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    halt();
    for (;;) {
        __asm__ volatile("nop");
    }
}
