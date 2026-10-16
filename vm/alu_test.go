// Package vm tests the encrypted RV64I ALU against plaintext instruction semantics.
//
// Fixtures build base-format instructions and compare encrypted ALU outputs.
// Register and immediate tables cover every funct3 result at reduced width.
// Separate pins exercise 64-bit arithmetic, signed comparisons, and LUI/I immediates.
// These tests catch selector and sign-extension errors before machine-level execution.
// Production decode/writeback remains oblivious and is covered by machine tests.
package vm

import (
	"testing"
)

// u64 converts a (possibly negative) signed constant to its uint64 bits
// without triggering Go constant-overflow checks.
func u64(v int64) uint64 { return uint64(v) }

// plaintext RV64 encoders (base R / I formats)
func encR(opcode, rd, rs1, rs2, f3, f7 uint32) uint32 {
	return (f7<<25)|(rs2<<20)|(rs1<<15)|(f3<<12)|(rd<<7)|opcode
}
// encI builds an RV64 I-type word with imm[11:0] in instruction bits 31:20.
func encI(opcode, rd, rs1, imm12, f3 uint32) uint32 {
	return ((imm12&0xFFF)<<20)|(rs1<<15)|(f3<<12)|(rd<<7)|opcode
}

// runALUCase executes one OP or OP-IMM instruction obliviously at width w
// and checks the writeback value against Go's RV64 semantics at that width.
func runALUCase(t *testing.T, m *Machine, w int, instWord uint32, rs1v, rs2v uint64, isImm bool, want uint64) {
	t.Helper()
	inst := &Instruction{W: m.EncryptWord(uint64(instWord), 32)}
	rs1 := m.EncryptWord(rs1v, w)
	rs2 := m.EncryptWord(rs2v, w)
	// immediate operand: sign-extend bits 20..31 of the instruction to width w
	imm := m.ZeroWord(w)
	for k := 0; k < w && k < 12; k++ {
		imm.Bits[k] = inst.bit(20 + k)
	}
	for k := 12; k < w; k++ {
		imm.Bits[k] = inst.bit(31)
	}
	flag := m.Ctx.EncryptBit(isImm)
	got := m.DecryptWord(m.ExecALU(inst, rs1, rs2, imm, flag))
	if got != uMask(want, w) {
		t.Errorf("inst=%#08x rs1=%#x rs2=%#x imm=%v: got %#x, want %#x",
			instWord, rs1v, rs2v, isImm, got, uMask(want, w))
	}
}

// TestALURegOps checks all eight OP funct3 choices and the register-form SUB/SRA selectors.
func TestALURegOps(t *testing.T) {
	m := testMachine(t)
	const w = 8
	type tc struct {
		f3, f7 uint32
		a, b   uint64
		want   uint64
		name   string
	}
	cases := []tc{
		{0x0, 0x00, 5, 7, 12, "ADD"},
		{0x0, 0x20, 5, 7, u64(5 - 7), "SUB"},
		{0x1, 0x00, 0x81, 3, uint64(0x81 << 3), "SLL"},
		{0x2, 0x00, 0xFF, 0x01, ^uint64(0), "SLT(-1<1)"},
		{0x2, 0x00, 0x7F, 0x80, 0, "SLT(127<-128)"},
		{0x3, 0x00, 0x01, 0xFF, ^uint64(0), "SLTU(1<255)"},
		{0x4, 0x00, 0xAA, 0x55, 0xFF, "XOR"},
		{0x6, 0x00, 0xAA, 0x55, 0xFF, "OR"},
		{0x7, 0x00, 0xAA, 0x55, 0x00, "AND"},
		{0x5, 0x00, 0x81, 3, 0x10, "SRL"},
		{0x5, 0x20, 0x81, 3, 0xF0, "SRA"},
	}
	for _, c := range cases {
		enc := encR(OpReg, 1, 2, 3, c.f3, c.f7)
		t.Run(c.name, func(t *testing.T) {
			runALUCase(t, m, w, enc, c.a, c.b, false, c.want)
		})
	}
}

// TestALUImmOps checks OP-IMM arithmetic, comparisons, and shift-immediate selectors.
func TestALUImmOps(t *testing.T) {
	m := testMachine(t)
	const w = 8
	type tc struct {
		f3   uint32
		sham uint32 // low 6 bits of imm field for shifts
		a    uint64
		imm  int32
		want uint64
		name string
	}
	cases := []tc{
		{0x0, 0, 5, -3, uint64(5 - 3), "ADDI"},
		{0x0, 0, 0x7F, 1, uint64(0x7F + 1), "ADDI-overflow"},
		{0x2, 0, 0x7F, -1, 0, "SLTI(127<-1)=0"},
		{0x3, 0, 0x01, -1, ^uint64(0), "SLTIU(1<255)=-1"}, // imm -1 unsigned = 255
		{0x1, 3, 0x01, 3, 0x08, "SLLI"},
		{0x5, 3 | (0 << 5), 0x81, 3, 0x10, "SRLI"},
		{0x5, 3 | (1 << 10), 0x81, 3, 0xF0, "SRAI"}, // SRAI selector = imm bit 10 (inst bit 30)
	}
	for _, c := range cases {
		imm12 := uint32(c.imm) & 0xFFF
		// for shifts the shamt lives in imm bits 0..5; SRAI sets imm bit 5
		if c.f3 == 0x1 || c.f3 == 0x5 {
			imm12 = c.sham & 0x3F
			if c.f3 == 0x5 && c.sham&(1<<10) != 0 {
				imm12 |= 1 << 10
			}
		}
		enc := encI(OpImm, 1, 2, imm12, c.f3)
		t.Run(c.name, func(t *testing.T) {
			runALUCase(t, m, w, enc, c.a, 0, true, c.want)
		})
	}
}

// full 64-bit path pins: one ADD, one SRA, one SLT at true RV64 width
func TestALU64BitPins(t *testing.T) {
	m := testMachine(t)
	const w = 64
	enc := encR(OpReg, 1, 2, 3, 0x0, 0x00)
	runALUCase(t, m, w, enc, 0xDEADBEEFCAFEBABE, 0x1234567890ABCDEF, false,
		0xDEADBEEFCAFEBABE+0x1234567890ABCDEF)
	enc = encR(OpReg, 1, 2, 3, 0x5, 0x20)
	nv := int64(-255)
	runALUCase(t, m, w, enc, 0xFFFFFFFFFFFFFF01, 4, false, uint64(nv>>4))
	enc = encR(OpReg, 1, 2, 3, 0x2, 0x00)
	runALUCase(t, m, w, enc, 0xFFFFFFFFFFFFFFFF, 1, false, ^uint64(0)) // -1 < 1 -> -1
}

// TestLUIAndImmI pins U-type LUI placement/sign extension and I-type sign extension.
func TestLUIAndImmI(t *testing.T) {
	m := testMachine(t)
	// LUI x1, 0x12345 -> 0x0000000012345000
	enc := (uint32(0x12345)<<12)|(1<<7)|OpLUI
	inst := &Instruction{W: m.EncryptWord(uint64(enc), 32)}
	if got := m.DecryptWord(m.LUIValue(inst)); got != uint64(0x0000000012345000) {
		t.Errorf("LUI 0x12345 = %#x", got)
	}
	// negative upper imm: LUI with 0xFFAAA -> sign-extended 0xFFFFFFFFFFAAA000
	enc = (uint32(0xFFAAA)<<12)|(1<<7)|OpLUI
	inst = &Instruction{W: m.EncryptWord(uint64(enc), 32)}
	// 0xFFAAA << 12 = 0xFFAAA000, sign-extended to 64 bits:
	if got := m.DecryptWord(m.LUIValue(inst)); got != uint64(0xFFFFFFFFFFAAA000) {
		t.Errorf("LUI 0xFFAAA = %#x, want %#x", got, uint64(0xFFFFFFFFFFAAA000))
	}
	// I-imm -1 -> all ones
	enc = encI(OpImm, 1, 2, 0xFFF, 0x0)
	inst = &Instruction{W: m.EncryptWord(uint64(enc), 32)}
	if got := m.DecryptWord(m.ImmI(inst)); got != 0xFFFFFFFFFFFFFFFF {
		t.Errorf("ImmI(-1) = %#x", got)
	}
	// I-imm -2048
	enc = encI(OpImm, 1, 2, 0x800, 0x0)
	inst = &Instruction{W: m.EncryptWord(uint64(enc), 32)}
	if got := m.DecryptWord(m.ImmI(inst)); got != u64(-2048) {
		t.Errorf("ImmI(-2048) = %#x", got)
	}
}
