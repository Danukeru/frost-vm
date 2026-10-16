// Package vm implements encrypted RV64I instruction decoding and ALU operations.
//
// This file defines opcode values, encrypted instruction fields, and ALU work.
// Field extraction is ciphertext wiring; constant matches and operations use gates.
// ExecALU computes every OP/OP-IMM candidate before funct3 selects its result.
// Step gates writeback by opcode and suppresses machine effects after halt.
// Fetch/decode-once, x0, and PC-freeze invariants are completed by the machine layer.
package vm

import (
	fhe "github.com/flashnext/frost-vm/fhe/bool"
)

// Op* names the 7-bit opcode values for the RV64I base instruction formats.
const (
	OpLUI    = 0b0110111
	OpAUIPC  = 0b0010111
	OpJAL    = 0b1101111
	OpJALR   = 0b1100111
	OpBranch = 0b1100011
	OpLoad   = 0b0000011
	OpStore  = 0b0100011
	OpImm    = 0b0010011 // OP-IMM
	OpFence  = 0b0001111
	OpSystem = 0b1110011 // ECALL/EBREAK
	OpReg    = 0b0110011 // OP (register-register)
)

// Instruction is an encrypted 32-bit RV64 instruction. Field access is
// free (bit references into the encrypted word); matching against
// constants costs only AND gates (constant bits use free NOTs).
type Instruction struct{ W *Word }

// bit returns the ciphertext at instruction bit index k.
func (i *Instruction) bit(k int) *fhe.LweSample { return i.W.Bits[k] }

// field returns the encrypted sub-word [lo, lo+n) (LSB-first).
func (i *Instruction) field(lo, n int) *Word {
	out := &Word{Bits: make([]*fhe.LweSample, n)}
	for k := 0; k < n; k++ {
		out.Bits[k] = i.W.Bits[lo+k]
	}
	return out
}

// Opcode / Rd / Funct3 / Rs1 / Rs2 / Shamt / Funct7Bit5 expose the standard
// RV64I fields; Shamt is six bits and Funct7Bit5 is the bit-30 OP selector.
func (i *Instruction) Opcode() *Word  { return i.field(0, 7) }
func (i *Instruction) Rd() *Word      { return i.field(7, 5) }
func (i *Instruction) Funct3() *Word  { return i.field(12, 3) }
func (i *Instruction) Rs1() *Word     { return i.field(15, 5) }
func (i *Instruction) Rs2() *Word     { return i.field(20, 5) }
func (i *Instruction) Shamt() *Word   { return i.field(20, 6) } // I-type shamt (6 bits for RV64)
func (i *Instruction) Funct7Bit5() *fhe.LweSample { return i.W.Bits[30] } // SUB / SRA selector

// MatchConst returns 1 iff bits [lo, lo+n) equal the constant val.
// Constant-0 bits compare via free NOT; only constant-1 bits cost an AND.
func (m *Machine) MatchConst(i *Instruction, lo, n int, val int) *fhe.LweSample {
	var acc *fhe.LweSample
	for k := 0; k < n; k++ {
		b := i.W.Bits[lo+k]
		if (val>>uint(k))&1 == 0 {
			b = m.Not(b)
		}
		if acc == nil {
			acc = b
		} else {
			acc = m.And(acc, b)
		}
	}
	if acc == nil {
		return m.oneBit
	}
	return acc
}

// ALUResult is the oblivious result of an OP / OP-IMM execution: every
// candidate operation is computed and the funct3 (plus funct7/shamt
// selectors) mux-selects. The caller gates the writeback by opcode match.
type ALUResult struct {
	Value *Word
}

// ExecALU computes the OP/OP-IMM result for this instruction given the
// already-selected register values. isOpImm selects the operand source
// (immediate vs rs2). shamt for OP comes from rs2 low 6 bits (RV64
// masks shamt to 6 bits); for OP-IMM it is the shamt field.
func (m *Machine) ExecALU(inst *Instruction, rs1, rs2 *Word, imm *Word, isOpImm *fhe.LweSample) *Word {
	// operand b: immediate for OP-IMM, rs2 for OP
	b := m.WordMux(isOpImm, imm, rs2)

	f3 := inst.Funct3()
	// shamt source: I-type shamt field vs rs2 low 6 bits (RV64 masks shamt
	// to 6 bits). Built per-bit so it works at any operand width.
	shW := 6
	if rs2.W() < shW {
		shW = rs2.W()
	}
	shamt := &Word{Bits: make([]*fhe.LweSample, shW)}
	for k := 0; k < shW; k++ {
		shamt.Bits[k] = m.Mux(isOpImm, inst.bit(20+k), rs2.Bits[k])
	}

	// candidate results for each funct3
	addRes := m.Add(rs1, b)
	subRes := m.Sub(rs1, b)
	// ADD/SUB selector: OP funct7 bit 5 (inst bit 30). For OP-IMM the same
	// bit belongs to the immediate (ADDI may legally have it set), so the
	// immediate form is always ADD.
	addSel := m.Mux(isOpImm, m.oneBit, m.MatchConst(inst, 30, 1, 0))
	addRes = m.WordMux(addSel, addRes, subRes)

	sltRes := m.LtS(rs1, b)
	sltuRes := m.LtU(rs1, b)
	xorRes := m.WordXor(rs1, b)
	orRes := m.WordOr(rs1, b)
	andRes := m.WordAnd(rs1, b)

	sllRes := m.ShiftLeft(rs1, shamt)
	srlRes := m.ShiftRightLogical(rs1, shamt)
	sraRes := m.ShiftRightArithmetic(rs1, shamt)
	srSel := m.MatchConst(inst, 30, 1, 0) // 0=SRL, 1=SRA
	srRes := m.WordMux(srSel, srlRes, sraRes)

	// widen single-bit results (SLT/SLTU) to a word: sign-extend the flag
	sltW := m.extendFlag(sltRes, rs1.W())
	sltuW := m.extendFlag(sltuRes, rs1.W())

	// 8-way mux tree over funct3
	c0 := addRes // 000
	c1 := sllRes // 001
	c2 := sltW   // 010
	c3 := sltuW  // 011
	c4 := xorRes // 100
	c5 := srRes  // 101
	c6 := orRes  // 110
	c7 := andRes // 111

	l0 := m.WordMux(f3.Bits[0], c1, c0)
	l1 := m.WordMux(f3.Bits[0], c3, c2)
	l2 := m.WordMux(f3.Bits[0], c5, c4)
	l3 := m.WordMux(f3.Bits[0], c7, c6)
	h0 := m.WordMux(f3.Bits[1], l1, l0)
	h1 := m.WordMux(f3.Bits[1], l3, l2)
	return m.WordMux(f3.Bits[2], h1, h0)
}

// extendFlag turns a 1-bit boolean into a w-bit word: 0 -> 0, 1 -> all ones
// (RV64 SLT/SLTI return 0/-1).
func (m *Machine) extendFlag(flag *fhe.LweSample, w int) *Word {
	out := &Word{Bits: make([]*fhe.LweSample, w)}
	ones := &Word{Bits: make([]*fhe.LweSample, w)}
	for i := range out.Bits {
		ones.Bits[i] = m.oneBit
	}
	zero := m.ZeroWord(w)
	return m.WordMux(flag, ones, zero)
}

// LUI value: the U-type immediate (bits 31..12) placed at bits 63..12 of a
// 64-bit word (sign-extended per RV64: bits 31..12 -> 63..12, and bits 63..32
// copy bit 31 for sign extension).
func (m *Machine) LUIValue(inst *Instruction) *Word {
	w := 64
	out := &Word{Bits: make([]*fhe.LweSample, w)}
	// bits 12..31 = instruction bits 12..31
	for k := 12; k < 32; k++ {
		out.Bits[k] = inst.bit(k)
	}
	// bits 0..11 = 0
	for k := 0; k < 12; k++ {
		out.Bits[k] = m.zeroBit
	}
	// bits 32..63 = sign extension of bit 31
	for k := 32; k < w; k++ {
		out.Bits[k] = inst.bit(31)
	}
	return out
}

// ImmI returns the sign-extended I-type immediate (bits 31..20) as a 64-bit
// encrypted word.
func (m *Machine) ImmI(inst *Instruction) *Word {
	out := &Word{Bits: make([]*fhe.LweSample, 64)}
	for k := 0; k < 12; k++ {
		out.Bits[k] = inst.bit(20 + k)
	}
	for k := 12; k < 64; k++ {
		out.Bits[k] = inst.bit(31)
	}
	return out
}
