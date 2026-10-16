// Package vm tests full oblivious machine programs assembled from RV64I words.
//
// Hand-encoded programs exercise multiple complete fetch/decode/merge cycles.
// Branch coverage includes both not-taken fallthrough and taken instruction skipping.
// A store/load program pins an encrypted-memory round trip and byte-lane effects.
// A JAL program checks the jump target and its pc+4 link-register value.
// Each program halts with ECALL; x0 and halted-state freezing remain machine invariants.
package vm

import (
	"testing"
)

// B/S/J-type plaintext encoders (RISC-V base formats)
func encB(imm int32, rs1, rs2, f3 uint32) uint32 {
	u := uint32(imm)
	inst := OpBranch | (f3 << 12) | (rs1 << 15) | (rs2 << 20)
	inst |= ((u >> 1) & 0xF) << 8 // imm[4:1] -> inst[11:8]
	inst |= ((u >> 5) & 0x3F) << 25
	inst |= ((u >> 11) & 1) << 7
	inst |= ((u >> 31) & 1) << 31
	return inst
}

// encS builds an RV64 S-type word with imm[11:5] at 31:25 and imm[4:0] at 11:7.
func encS(imm int32, rs1, rs2, f3 uint32) uint32 {
	u := uint32(imm)
	inst := OpStore | (f3 << 12) | (rs1 << 15) | (rs2 << 20)
	inst |= (u & 0x1F) << 7 // imm[4:0] -> inst[11:7]
	inst |= ((u >> 5) & 0x3F) << 25
	inst |= ((u >> 31) & 1) << 31
	return inst
}

// encJ builds an RV64 J-type JAL word with the split immediate and implicit bit 0.
func encJ(imm int32, rd uint32) uint32 {
	u := uint32(imm)
	inst := OpJAL | (rd << 7)
	inst |= ((u >> 1) & 0x3FF) << 21 // imm[10:1] -> inst[30:21]
	inst |= ((u >> 11) & 1) << 20
	inst |= ((u >> 12) & 0xFF) << 12 // imm[19:12] -> inst[19:12]
	inst |= ((u >> 31) & 1) << 31
	return inst
}

// newTestRV creates a test machine with the requested power-of-two geometry and loads prog.
func newTestRV(t *testing.T, nProg, nRegs, nMem int, prog []uint32) *RVMachine {
	t.Helper()
	rv, err := NewRVMachine(testMachine(t).Ctx, nProg, nRegs, nMem)
	if err != nil {
		t.Fatalf("machine: %v", err)
	}
	if err := rv.LoadProgram(prog); err != nil {
		t.Fatalf("program: %v", err)
	}
	return rv
}

// Program A: OP-IMM + OP + branches in both directions.
//
//	0: ADDI x1,x0,5
//	4: ADDI x2,x0,7
//	8: BEQ x1,x2,+8   not taken (5!=7) -> fall through
//	12: ADDI x3,x1,37 -> x3 = 42
//	16: BEQ x1,x1,+8  taken -> skip next
//	20: ADDI x3,x0,1  (skipped)
//	24: ECALL
func TestMachineALUAndBranches(t *testing.T) {
	prog := []uint32{
		encI(OpImm, 1, 0, 5, 0x0),
		encI(OpImm, 2, 0, 7, 0x0),
		encB(8, 1, 2, F3BEQ),
		encI(OpImm, 3, 1, 37, 0x0),
		encB(8, 1, 1, F3BEQ),
		encI(OpImm, 3, 0, 1, 0x0),
		encI(OpSystem, 0, 0, 0, 0),
	}
	rv := newTestRV(t, 8, 4, 8, prog)
	steps := rv.Run(10)
	t.Logf("ran %d steps, pc=%d halted=%v", steps, rv.DecryptPC(), rv.DecryptHalted())
	if !rv.DecryptHalted() {
		t.Errorf("machine did not halt")
	}
	if got := rv.DecryptReg(1); got != 5 {
		t.Errorf("x1 = %d, want 5", got)
	}
	if got := rv.DecryptReg(2); got != 7 {
		t.Errorf("x2 = %d, want 7", got)
	}
	if got := rv.DecryptReg(3); got != 42 {
		t.Errorf("x3 = %d, want 42 (not-taken path must execute ADDI x3,x1,37 and the taken branch must skip ADDI x3,x0,1)", got)
	}
}

// Program B: SD then LD round-trip through encrypted memory.
//
//	0: ADDI x1,x0,77
//	4: ADDI x2,x0,3
//	8: SD x1,0(x2)
//	12: ADDI x3,x0,0
//	16: LD x3,0(x2) -> x3 = 77
//	20: ECALL
func TestMachineLoadStore(t *testing.T) {
	prog := []uint32{
		encI(OpImm, 1, 0, 77, 0x0),
		encI(OpImm, 2, 0, 3, 0x0),
		encS(0, 2, 1, F3SD),
		encI(OpImm, 3, 0, 0, 0x0),
		encI(OpLoad, 3, 2, 0, F3LD),
		encI(OpSystem, 0, 0, 0, 0),
	}
	rv := newTestRV(t, 8, 4, 8, prog)
	steps := rv.Run(10)
	t.Logf("ran %d steps", steps)
	if got := rv.DecryptReg(3); got != 77 {
		t.Errorf("x3 = %d, want 77 (SD/LD round trip)", got)
	}
	// byte-level check: cell 3 holds 77, cell 4 holds 0 (SD wrote 8 bytes)
	if got := rv.DecryptMemByte(3); got != 77 {
		t.Errorf("mem[3] = %d, want 77", got)
	}
	if got := rv.DecryptMemByte(4); got != 0 {
		t.Errorf("mem[4] = %d, want 0", got)
	}
}

// Program C: JAL (taken jump + link).
//
//	0: JAL x1,+8   -> x1 = 4, pc = 8
//	4: ADDI x2,x0,1 (skipped)
//	8: ADDI x3,x0,2
//	12: ECALL
func TestMachineJAL(t *testing.T) {
	prog := []uint32{
		encJ(8, 1),
		encI(OpImm, 2, 0, 1, 0x0),
		encI(OpImm, 3, 0, 2, 0x0),
		encI(OpSystem, 0, 0, 0, 0),
	}
	rv := newTestRV(t, 4, 4, 8, prog)
	rv.Run(6)
	if got := rv.DecryptReg(1); got != 4 {
		t.Errorf("x1 = %d, want 4 (JAL link = pc+4)", got)
	}
	if got := rv.DecryptReg(2); got != 0 {
		t.Errorf("x2 = %d, want 0 (instruction after JAL must be skipped)", got)
	}
	if got := rv.DecryptReg(3); got != 2 {
		t.Errorf("x3 = %d, want 2", got)
	}
}
