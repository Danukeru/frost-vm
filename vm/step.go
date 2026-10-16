// Package vm implements one full oblivious RV64 execution cycle per Step call.
//
// Step fetches, decodes, executes all instruction candidates, then merges state.
// PC and register indices are decoded once and selected with masked OR.
// Candidate execution never decrypts the control state or chooses a clear path.
// x0 is never written, and halt-gated effects make the halted machine a fixed point.
// The PC is explicitly held at its prior encrypted value while halted.
package vm

import (
	"log"

	fhe "github.com/flashnext/frost-vm/fhe/bool"
)

// decode returns flags[i] = 1 iff w == i, for i in [0,n). Constant-bit
// comparisons use free NOTs; only set bits of i cost an AND.
func (rv *RVMachine) decode(w *Word, n int) []*fhe.LweSample {
	out := make([]*fhe.LweSample, n)
	for i := 0; i < n; i++ {
		out[i] = rv.matchConstWord(w, uint64(i))
	}
	return out
}

// orAll OR-reduces gate outputs.
func (rv *RVMachine) orAll(bits []*fhe.LweSample) *fhe.LweSample {
	acc := bits[0]
	for _, b := range bits[1:] {
		acc = rv.Or(acc, b)
	}
	return acc
}

// Step executes one oblivious machine cycle. Every candidate slot and
// instruction effect is computed; selection is by masked OR (AND/OR decode,
// ~4x cheaper than per-slot mux chains) for the RAM and register-file
// reads, and by mux for the few-bit control selections (PC, ALU funct3,
// load extension). When the machine is halted, all effects are gated off
// (the cycle is a no-op).
func (rv *RVMachine) Step() {
	// ---- fetch: instruction bit = OR over program slots of (PC match & bit) ----
	decPC := make([]*fhe.LweSample, rv.nProg)
	for i := 0; i < rv.nProg; i++ {
		decPC[i] = rv.matchConstWord(rv.pc, uint64(4*i)) // byte addresses
	}
	instW := &Word{Bits: make([]*fhe.LweSample, 32)}
	for k := 0; k < 32; k++ {
		terms := make([]*fhe.LweSample, rv.nProg)
		for i := 0; i < rv.nProg; i++ {
			terms[i] = rv.And(decPC[i], rv.insts[i].Bits[k])
		}
		instW.Bits[k] = rv.orAll(terms)
	}
	I := &Instruction{W: instW}
	notHalted := rv.Not(rv.halted)

	// ---- opcode match flags (gated by not-halted) ----
	isOp := rv.andFlags(rv.MatchConst(I, 0, 7, OpReg), notHalted)
	isOpImm := rv.andFlags(rv.MatchConst(I, 0, 7, OpImm), notHalted)
	isLUI := rv.andFlags(rv.MatchConst(I, 0, 7, OpLUI), notHalted)
	isAUIPC := rv.andFlags(rv.MatchConst(I, 0, 7, OpAUIPC), notHalted)
	isJAL := rv.andFlags(rv.MatchConst(I, 0, 7, OpJAL), notHalted)
	isJALR := rv.andFlags(rv.MatchConst(I, 0, 7, OpJALR), notHalted)
	isBranch := rv.andFlags(rv.MatchConst(I, 0, 7, OpBranch), notHalted)
	isLoad := rv.andFlags(rv.MatchConst(I, 0, 7, OpLoad), notHalted)
	isStore := rv.andFlags(rv.MatchConst(I, 0, 7, OpStore), notHalted)
	// ECALL: SYSTEM opcode with funct3=0 and imm=0 (bits 12..31 all zero)
	isEcall := rv.andFlags(rv.MatchConst(I, 0, 7, OpSystem), rv.MatchConst(I, 12, 20, 0))
	isEcall = rv.andFlags(isEcall, notHalted)

	// ---- register reads: value bit = OR over file of (field match & bit) ----
	decRs1 := make([]*fhe.LweSample, rv.nRegs)
	decRs2 := make([]*fhe.LweSample, rv.nRegs)
	decRd := make([]*fhe.LweSample, rv.nRegs)
	for r := 0; r < rv.nRegs; r++ {
		decRs1[r] = rv.MatchConst(I, 15, 5, r)
		decRs2[r] = rv.MatchConst(I, 20, 5, r)
		decRd[r] = rv.MatchConst(I, 7, 5, r)
	}
	readRegFile := func(dec []*fhe.LweSample) *Word {
		out := &Word{Bits: make([]*fhe.LweSample, 64)}
		for k := 0; k < 64; k++ {
			terms := make([]*fhe.LweSample, rv.nRegs)
			for r := 0; r < rv.nRegs; r++ {
				terms[r] = rv.And(dec[r], rv.regs[r].Bits[k])
			}
			out.Bits[k] = rv.orAll(terms)
		}
		return out
	}
	rs1v := readRegFile(decRs1)
	rs2v := readRegFile(decRs2)

	// ---- immediates (free bit wiring) ----
	immI := rv.ImmI(I)
	immS := rv.ImmS(I)
	immB := rv.ImmB(I)
	immJ := rv.ImmJ(I)

	// ---- execute all candidates ----
	aluVal := rv.ExecALU(I, rs1v, rs2v, immI, isOpImm)
	luiVal := rv.LUIValue(I)

	pcExt := rv.ZeroWord(64)
	for k := 0; k < rv.pcW; k++ {
		pcExt.Bits[k] = rv.pc.Bits[k]
	}
	auipcVal := rv.Add(pcExt, luiVal)

	pcPlus4 := rv.Add(rv.pc, rv.ConstWord(4, rv.pcW))
	pcPlus4Ext := rv.ZeroWord(64)
	for k := 0; k < rv.pcW; k++ {
		pcPlus4Ext.Bits[k] = pcPlus4.Bits[k]
	}

	// effective addresses: loads/JALR use rs1+immI; stores use rs1+immS
	addr64 := rv.Add(rs1v, immI)
	addrLow := &Word{Bits: addr64.Bits[:rv.memW]}
	addrS64 := rv.Add(rs1v, immS)
	addrSLow := &Word{Bits: addrS64.Bits[:rv.memW]}

	// branch condition mux over funct3: 000 BEQ, 001 BNE, 100 BLT,
	// 101 BGE, 110 BLTU, 111 BGEU; 010/011 reserved (never taken).
	eq := rv.Eq(rs1v, rs2v)
	lts := rv.LtS(rs1v, rs2v)
	ltu := rv.LtU(rs1v, rs2v)
	f3 := I.Funct3()
	t0 := rv.Mux(f3.Bits[0], rv.Not(eq), eq)
	t2 := rv.Mux(f3.Bits[0], rv.Not(lts), lts)
	t6 := rv.Mux(f3.Bits[0], rv.Not(ltu), ltu)
	a01 := rv.Mux(f3.Bits[1], rv.zeroBit, t0)
	a23 := rv.Mux(f3.Bits[1], t6, t2)
	taken := rv.Mux(f3.Bits[2], a23, a01)
	branchTaken := rv.andFlags(isBranch, taken)

	// ---- load: address decode once; value bit = OR of (cell match & bit) ----
	decA := rv.decode(addrLow, rv.nMem)
	loaded := rv.ZeroWord(64)
	for j := 0; j < 8; j++ {
		for k := 0; k < 8; k++ {
			terms := make([]*fhe.LweSample, rv.nMem)
			for c := 0; c < rv.nMem; c++ {
				terms[c] = rv.And(decA[(c-j+rv.nMem)%rv.nMem], rv.mem[c].Bits[k])
			}
			loaded.Bits[8*j+k] = rv.orAll(terms)
		}
	}
	// sign/zero extension per funct3. Only bits at or above the access
	// width are replaced; bits below stay real data. (A previous version
	// replaced everything >= bit 8 for all widths, silently dropping real
	// data bits 8..31 on LW/LHU/LWU and 8..15 on LH/LBU-style loads:
	// LW of any value >= 256 lost its bit 8.)
	s1 := loaded.Bits[7]  // LB sign
	s2 := loaded.Bits[15] // LH sign
	s4 := loaded.Bits[31] // LW sign
	for k := 8; k < 64; k++ {
		// candidate per funct3 value: 000 LB, 001 LH, 010 LW, 011 LD,
		// 100 LBU, 101 LHU, 110 LWU, 111 (reserved: keep real bits)
		c0 := s1
		c1 := s2
		c2 := s4
		c3 := loaded.Bits[k] // LD: real data at every k
		c4 := rv.zeroBit
		c5 := rv.zeroBit
		c6 := rv.zeroBit
		c7 := loaded.Bits[k]
		if k < 16 {
			c1 = loaded.Bits[k] // LH: bits 8..15 are real data
			c5 = loaded.Bits[k] // LHU: same
		}
		if k < 32 {
			c2 = loaded.Bits[k] // LW: bits 8..31 are real data
			c6 = loaded.Bits[k] // LWU: same
		}
		l0 := rv.Mux(f3.Bits[0], c1, c0)
		l1 := rv.Mux(f3.Bits[0], c3, c2)
		l2 := rv.Mux(f3.Bits[0], c5, c4)
		l3 := rv.Mux(f3.Bits[0], c7, c6)
		h0 := rv.Mux(f3.Bits[1], l1, l0)
		h1 := rv.Mux(f3.Bits[1], l3, l2)
		loaded.Bits[k] = rv.Mux(f3.Bits[2], h1, h0)
	}

	// ---- store: per cell, write-enable = OR_j (addr+j match & size enable) ----
	// store sizes: SB=1, SH=2, SW=4, SD=8 (funct3 000,001,010,011).
	sizeEn := make([]*fhe.LweSample, 8) // includes isStore gating
	{
		m0 := rv.MatchConst(I, 12, 3, F3SB)
		m1 := rv.MatchConst(I, 12, 3, F3SH)
		m2 := rv.MatchConst(I, 12, 3, F3SW)
		m3 := rv.MatchConst(I, 12, 3, F3SD)
		all := rv.orFlags(rv.orFlags(m0, m1), rv.orFlags(m2, m3))
		shOrAbove := rv.orFlags(m1, rv.orFlags(m2, m3))
		swOrAbove := rv.orFlags(m2, m3)
		sizeEn[0] = rv.andFlags(isStore, all)
		sizeEn[1] = rv.andFlags(isStore, shOrAbove)
		sizeEn[2] = rv.andFlags(isStore, swOrAbove)
		sizeEn[3] = rv.andFlags(isStore, swOrAbove)
		for j := 4; j < 8; j++ {
			sizeEn[j] = rv.andFlags(isStore, m3)
		}
	}
	decS := rv.decode(addrSLow, rv.nMem)
	for c := 0; c < rv.nMem; c++ {
		wj := make([]*fhe.LweSample, 8)
		for j := 0; j < 8; j++ {
			wj[j] = rv.And(decS[(c-j+rv.nMem)%rv.nMem], sizeEn[j])
		}
		any := rv.orAll(wj)
		keep := rv.Not(any)
		newCell := &Word{Bits: make([]*fhe.LweSample, 8)}
		for k := 0; k < 8; k++ {
			terms := make([]*fhe.LweSample, 8)
			for j := 0; j < 8; j++ {
				terms[j] = rv.And(wj[j], rs2v.Bits[8*j+k])
			}
			dataOr := rv.orAll(terms)
			newCell.Bits[k] = rv.Or(rv.And(keep, rv.mem[c].Bits[k]), dataOr)
		}
		rv.mem[c] = newCell
	}

	// ---- writeback value: mux over instruction classes ----
	rdCur := readRegFile(decRd)
	jalRd := rv.orFlags(isJAL, isJALR)
	aluFlag := rv.orFlags(isOp, isOpImm)
	writeEn := rv.orFlags(aluFlag, rv.orFlags(isLUI, rv.orFlags(isAUIPC, rv.orFlags(jalRd, isLoad))))

	val := rdCur
	val = rv.WordMux(isLoad, loaded, val)
	val = rv.WordMux(jalRd, pcPlus4Ext, val)
	val = rv.WordMux(isAUIPC, auipcVal, val)
	val = rv.WordMux(isLUI, luiVal, val)
	val = rv.WordMux(aluFlag, aluVal, val)

	// register write: masked update (2 AND + 1 OR per bit, cheaper than mux)
	for r := 1; r < rv.nRegs; r++ { // x0 (index 0) is never written
		wr := rv.andFlags(decRd[r], writeEn)
		nw := rv.Not(wr)
		updated := &Word{Bits: make([]*fhe.LweSample, 64)}
		for k := 0; k < 64; k++ {
			updated.Bits[k] = rv.Or(rv.And(wr, val.Bits[k]), rv.And(nw, rv.regs[r].Bits[k]))
		}
		rv.regs[r] = updated
	}

	// ---- PC update ----
	pcNext := pcPlus4
	pcNext = rv.WordMux(branchTaken, rv.Add(rv.pc, immB), pcNext)
	pcNext = rv.WordMux(isJAL, rv.Add(rv.pc, immJ), pcNext)
	jalrT := &Word{Bits: make([]*fhe.LweSample, rv.pcW)}
	jalrT.Bits[0] = rv.zeroBit // JALR clears bit 0
	for k := 1; k < rv.pcW; k++ {
		jalrT.Bits[k] = addr64.Bits[k]
	}
	pcNext = rv.WordMux(isJALR, jalrT, pcNext)
	rv.pc = rv.WordMux(rv.halted, rv.pc, pcNext) // frozen when halted

	// ---- halt ----
	rv.halted = rv.Or(rv.halted, isEcall)
}

// Run executes up to maxSteps cycles, stopping early when the (external
// observer's view of the) machine has halted. Progress goes to the log
// every 10 cycles (runs can take hours of real TFHE).
func (rv *RVMachine) Run(maxSteps int) int {
	steps := 0
	for i := 0; i < maxSteps; i++ {
		if rv.DecryptHalted() {
			break
		}
		rv.Step()
		steps++
		if steps%10 == 0 {
			log.Printf("frost-vm: %d cycles executed (pc=%d)", steps, rv.DecryptPC())
		}
	}
	return steps
}
