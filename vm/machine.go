// Package vm implements the encrypted RV64 machine state and its geometry.
//
// RVMachine holds encrypted PC, instruction slots, registers, byte RAM, and halt.
// Each cycle decodes addresses once and uses masked OR for slot selection.
// Power-of-two geometry masks PC and data addresses to the configured machine.
// Register x0 is hard-wired to zero; a halted cycle freezes the machine state.
// This file initializes state and exposes decryption only to external observers.
package vm

import (
	"fmt"

	fhe "github.com/flashnext/frost-vm/fhe/bool"
)

// RV64 funct3 encodings for branches / loads / stores.
const (
	// branches
	F3BEQ, F3BNE, F3BLT, F3BGE, F3BLTU, F3BGEU = 0x0, 0x1, 0x4, 0x5, 0x6, 0x7
	// loads
	F3LB, F3LH, F3LW, F3LD, F3LBU, F3LHU, F3LWU = 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6
	// stores (funct3 low 3 bits; bit 2 selects store opcode)
	F3SB, F3SH, F3SW, F3SD = 0x0, 0x1, 0x2, 0x3
)

// RVMachine is an oblivious RV64 machine: program, registers, data memory,
// PC and halt flag are all encrypted. Every cycle evaluates every
// candidate slot/instruction and mux-merges the effects (oblivious model).
//
// Machine parameters (nProg/nRegs/nMem) may be smaller than the ISA
// address space; unimplemented register slots read as zero and never
// match a fetch/store address, which is a sound subset machine. x0 is
// hard-wired zero (index 0 is never written).
type RVMachine struct {
	*Machine // word ops + FHE context

	nProg int
	pcW   int
	nRegs int
	nMem  int // bytes
	memW  int

	pc      *Word
	insts   []*Word
	regs    []*Word
	mem     []*Word // byte cells
	halted  *fhe.LweSample
}

// NewRVMachine builds a machine with an encrypted program of nProg
// 32-bit instructions, nRegs 64-bit registers and nMem bytes of data
// memory (all powers of two; nMem multiple of 8).
func NewRVMachine(ctx *fhe.Context, nProg, nRegs, nMem int) (*RVMachine, error) {
	if nProg&(nProg-1) != 0 || nRegs <= 0 || nRegs > 32 || nMem&(nMem-1) != 0 || nMem%8 != 0 {
		return nil, fmt.Errorf("vm: bad machine params nProg=%d nRegs=%d nMem=%d", nProg, nRegs, nMem)
	}
	rv := &RVMachine{
		Machine: NewMachine(ctx),
		nProg:   nProg, pcW: log2(nProg) + 2, // PC is a byte address (4 bytes/insn)
		nRegs: nRegs,
		nMem:  nMem, memW: log2(nMem),
	}
	rv.Reset()
	return rv, nil
}

// log2 returns the exponent for a positive power-of-two machine dimension.
func log2(n int) int {
	l := 0
	for 1<<uint(l) < n {
		l++
	}
	return l
}

// Reset clears PC, registers, memory and the halt flag. The program must
// be loaded separately (LoadProgram).
func (rv *RVMachine) Reset() {
	rv.pc = rv.EncryptWord(0, rv.pcW)
	rv.regs = make([]*Word, rv.nRegs)
	for i := range rv.regs {
		rv.regs[i] = rv.ZeroWord(64)
	}
	rv.mem = make([]*Word, rv.nMem)
	for i := range rv.mem {
		rv.mem[i] = rv.ZeroWord(8)
	}
	rv.halted = rv.Ctx.EncryptBit(false)
}

// LoadProgram encrypts the program (RV64 32-bit words, little-endian
// instruction memory at byte addresses 4*i).
func (rv *RVMachine) LoadProgram(prog []uint32) error {
	if len(prog) > rv.nProg {
		return fmt.Errorf("vm: program of %d instructions exceeds nProg=%d", len(prog), rv.nProg)
	}
	rv.insts = make([]*Word, rv.nProg)
	for i := 0; i < rv.nProg; i++ {
		var v uint64
		if i < len(prog) {
			v = uint64(prog[i])
		} // padding: zero word (never fetched in a well-formed run)
		rv.insts[i] = rv.EncryptWord(v, 32)
	}
	return nil
}

// DecryptReg / DecryptPC / DecryptHalted expose state to an external
// observer (the test oracle).
func (rv *RVMachine) DecryptReg(i int) uint64 { return rv.DecryptWord(rv.regs[i]) }
func (rv *RVMachine) DecryptPC() uint64       { return rv.DecryptWord(rv.pc) }
func (rv *RVMachine) DecryptHalted() bool     { return rv.Ctx.DecryptBit(rv.halted) }
// DecryptMemByte returns one decrypted RAM byte to the test oracle.
func (rv *RVMachine) DecryptMemByte(i int) uint64 { return rv.DecryptWord(rv.mem[i]) }

// matchConstWord returns 1 iff w equals the constant v (free NOTs for
// zero bits; ANDs only for one bits).
func (rv *RVMachine) matchConstWord(w *Word, v uint64) *fhe.LweSample {
	var acc *fhe.LweSample
	for k := 0; k < w.W(); k++ {
		b := w.Bits[k]
		if (v>>uint(k))&1 == 0 {
			b = rv.Not(b)
		}
		if acc == nil {
			acc = b
		} else {
			acc = rv.And(acc, b)
		}
	}
	if acc == nil {
		return rv.oneBit
	}
	return acc
}

// orFlags combines two encrypted effect or match flags with the OR gate.
func (rv *RVMachine) orFlags(a, b *fhe.LweSample) *fhe.LweSample { return rv.Or(a, b) }

// andFlags gates an encrypted effect or match flag with the AND gate.
func (rv *RVMachine) andFlags(a, b *fhe.LweSample) *fhe.LweSample { return rv.And(a, b) }

// ---------------------------------------------------------------------------
// immediates (bit wiring only — free)
// ---------------------------------------------------------------------------

// ImmB: sign-extended B-type immediate, truncated to pcW bits (mod 2^pcW
// wraparound arithmetic is correct for a power-of-two address space).
// RISC-V mapping: imm[12]=inst[31], imm[11]=inst[7], imm[10:5]=inst[30:25],
// imm[4:1]=inst[11:8], imm[0]=0.
func (rv *RVMachine) ImmB(inst *Instruction) *Word {
	out := &Word{Bits: make([]*fhe.LweSample, rv.pcW)}
	for k := 0; k < rv.pcW; k++ {
		switch {
		case k == 0:
			out.Bits[k] = rv.zeroBit
		case k <= 4:
			out.Bits[k] = inst.bit(7 + k) // imm[4:1] = inst[11:8] (imm[k]=inst[7+k])
		case k <= 10:
			out.Bits[k] = inst.bit(20 + k) // imm[10:5] = inst[30:25] (imm[k]=inst[20+k])
		case k == 11:
			out.Bits[k] = inst.bit(7) // imm[11]
		default:
			out.Bits[k] = inst.bit(31) // imm[12] and sign extension
		}
	}
	return out
}

// ImmS: sign-extended S-type immediate (store offset), 64-bit word.
// imm[4:0]=inst[7..11], imm[10:5]=inst[25..30], sign=inst[31].
func (rv *RVMachine) ImmS(inst *Instruction) *Word {
	out := &Word{Bits: make([]*fhe.LweSample, 64)}
	for k := 0; k < 5; k++ {
		out.Bits[k] = inst.bit(7 + k) // imm[4:0] = inst[7..11]
	}
	for k := 5; k < 12; k++ {
		out.Bits[k] = inst.bit(20 + k) // imm[10:5] = inst[25..30]
	}
	for k := 12; k < 64; k++ {
		out.Bits[k] = inst.bit(31) // sign
	}
	return out
}

// ImmJ: sign-extended J-type immediate (JAL offset), truncated to pcW bits.
// RISC-V mapping: imm[20]=inst[31], imm[19:12]=inst[19:12], imm[11]=inst[20],
// imm[10:1]=inst[30:21], imm[0]=0.
func (rv *RVMachine) ImmJ(inst *Instruction) *Word {
	out := &Word{Bits: make([]*fhe.LweSample, rv.pcW)}
	for k := 0; k < rv.pcW; k++ {
		switch {
		case k == 0:
			out.Bits[k] = rv.zeroBit
		case k <= 10:
			out.Bits[k] = inst.bit(20 + k) // imm[10:1] = inst[30:21] (imm[k]=inst[20+k])
		case k == 11:
			out.Bits[k] = inst.bit(20) // imm[11]
		case k <= 19:
			out.Bits[k] = inst.bit(k) // imm[19:12] = inst[19:12]
		default:
			out.Bits[k] = inst.bit(31) // imm[20] and sign extension
		}
	}
	return out
}
