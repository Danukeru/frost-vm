// Package vm contains ELF-loader and end-to-end Fibonacci machine tests.
//
// TestLoadELF verifies clang-emitted instruction encodings at known program slots.
// TestFibonacciProgram is the flagship compiled-C execution and memory-output check.
// That expensive test is gated by FROST_RUN_FIB=1 and self-halts through ECALL.
// Both tests use encrypted machine state; plaintext inspection is limited to assertions.
// Run applies the normal oblivious cycle, including masked-OR selection and halt freeze.
package vm

import (
	"os"
	"testing"
)

// TestLoadELF checks the loader geometry without running the machine:
// entry/segments validated, instruction slots decrypted back to the exact
// encodings clang emitted.
func TestLoadELF(t *testing.T) {
	const path = "../programs/fib.elf"
	if _, err := os.Stat(path); err != nil {
		t.Skip("build programs/fib.elf first (programs/build.sh):", err)
	}
	rv, err := NewRVMachine(testMachine(t).Ctx, 32, 16, 128)
	if err != nil {
		t.Fatal(err)
	}
	if err := rv.LoadELF(path); err != nil {
		t.Fatal(err)
	}
	// first instruction: lui a0, 0x1 (0x00001537); the add at slot 7
	// (0x1c); ecall at slot 9 (0x24). (For N=3 clang unrolls the loop.)
	if got := rv.DecryptWord(rv.insts[0]); got != 0x00001537 {
		t.Errorf("inst[0] = %#08x, want 0x00001537", got)
	}
	if got := rv.DecryptWord(rv.insts[7]); got != 0x00b50533 {
		t.Errorf("inst[7] = %#08x, want 0x00b50533 (add)", got)
	}
	if got := rv.DecryptWord(rv.insts[9]); got != 0x00000073 {
		t.Errorf("inst[9] = %#08x, want 0x00000073 (ecall)", got)
	}
}

// TestFibonacciProgram runs the clang-compiled RV64 fibonacci program
// end-to-end on the oblivious machine and checks the decrypted values:
// fib[0..2] = 0,1,1 and the rest of the array left zero.
//
// This is the flagship end-to-end test: ~10 real TFHE cycles (~40 min at
// machine test parameters). It is skipped unless FROST_RUN_FIB=1 so the
// default suite stays fast.
func TestFibonacciProgram(t *testing.T) {
	if os.Getenv("FROST_RUN_FIB") == "" {
		t.Skip("set FROST_RUN_FIB=1 to run the full fibonacci program (hours of real TFHE)")
	}
	const path = "../programs/fib.elf"
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("build programs/fib.elf first (programs/build.sh): %v", err)
	}
	// machine sized generously: 32 program slots, 16 registers, 128-byte RAM.
	rv, err := NewRVMachine(testMachine(t).Ctx, 32, 16, 128)
	if err != nil {
		t.Fatal(err)
	}
	if err := rv.LoadELF(path); err != nil {
		t.Fatal(err)
	}
	steps := rv.Run(50)
	t.Logf("fibonacci program: ran %d cycles, halted=%v", steps, rv.DecryptHalted())
	if !rv.DecryptHalted() {
		t.Errorf("machine did not halt within 50 cycles")
	}
	want := []uint64{0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	got := make([]uint64, 20)
	for i := range got {
		var v uint64
		for b := 0; b < 4; b++ {
			v |= rv.DecryptMemByte(i*4 + b) << (8 * uint(b)) // fib at 0x1000 -> mem[0..]
		}
		got[i] = v
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("fib[%d] = %d, want %d", i, got[i], want[i])
		}
	}
	t.Logf("fib[0..19] = %v", got)
}
