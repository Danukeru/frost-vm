// Package vm implements the oblivious RV64 virtual machine: the whole
// machine state (PC, registers, memory, instruction stream) is encrypted,
// and every cycle every candidate slot/instruction is evaluated and
// mux-merged — the oblivious-VM execution model, mapped from
// encrypted-Brainfuck FHE VMs to the RV64I base ISA.
//
// This file is the foundation layer: encrypted multi-bit words built from
// the fhe/bool boolean gate set. Bit order is little-endian: index 0 is
// the LSB (bit 0 in RV64 terms), which keeps adders and shifters direct.
//
// Gate-contract note (fhebool.Context): every bit of every Word handled
// here is either an EncryptBit/GateConstant output or a gate output, so
// the ±1/8 message space with noise < 1/16 invariant holds throughout.
//
// Word carries little-endian ciphertext bits and implements RV64 word gates.
// Arithmetic, comparisons, shifts, and muxes compose fhe-bool gate outputs.
// Every result stays in the gate layer's valid ciphertext message space.
// Step uses these operations to evaluate candidates before masked-OR merge.
// x0 and halted-state freezing are enforced by the machine execution layer.
package vm

import (
	"fmt"

	fhe "github.com/flashnext/frost-vm/fhe/bool"
)

// Machine owns the FHE context (keys) and shared constants. All Words
// handled by one Machine belong to that Machine.
type Machine struct {
	Ctx *fhe.Context

	zeroBit *fhe.LweSample // cached encrypted 0 (GateConstant output)
	oneBit  *fhe.LweSample // cached encrypted 1
}

// NewMachine builds a machine over an explicit FHE context.
func NewMachine(ctx *fhe.Context) *Machine {
	m := &Machine{Ctx: ctx}
	m.zeroBit = ctx.EncryptBit(false)
	m.oneBit = ctx.EncryptBit(true)
	return m
}

// Word is an encrypted w-bit unsigned value (little-endian bit vector).
type Word struct {
	Bits []*fhe.LweSample
}

// W returns the word width in bits.
func (w *Word) W() int { return len(w.Bits) }

// Clone returns a deep copy (fresh ciphertext objects, same values).
func (m *Machine) Clone(w *Word) *Word {
	out := &Word{Bits: make([]*fhe.LweSample, w.W())}
	for i, b := range w.Bits {
		out.Bits[i] = b.Copy()
	}
	return out
}

// EncryptWord encrypts a plaintext value into a w-bit word (w in [1,64]).
func (m *Machine) EncryptWord(v uint64, w int) *Word {
	out := &Word{Bits: make([]*fhe.LweSample, w)}
	for i := 0; i < w; i++ {
		out.Bits[i] = m.Ctx.EncryptBit((v>>uint(i))&1 == 1)
	}
	return out
}

// ConstWord returns an encrypted constant word using trivial encryptions
// (cheaper and lower-noise than EncryptBit for constants).
func (m *Machine) ConstWord(v uint64, w int) *Word {
	out := &Word{Bits: make([]*fhe.LweSample, w)}
	for i := 0; i < w; i++ {
		out.Bits[i] = m.Ctx.NewCiphertext()
		m.Ctx.GateConstant(out.Bits[i], (v>>uint(i))&1 == 1)
	}
	return out
}

// DecryptWord decrypts a word to its plaintext value.
func (m *Machine) DecryptWord(w *Word) uint64 {
	var v uint64
	for i, b := range w.Bits {
		if m.Ctx.DecryptBit(b) {
			v |= 1 << uint(i)
		}
	}
	return v
}

// ZeroBit / OneBit return cached encrypted constants. They must be treated
// as read-only inputs (gates never mutate their inputs).
func (m *Machine) ZeroBit() *fhe.LweSample { return m.zeroBit }
func (m *Machine) OneBit() *fhe.LweSample  { return m.oneBit }

// ---------------------------------------------------------------------------
// single-bit helpers
// ---------------------------------------------------------------------------

// gate allocates a fresh ciphertext and applies the supplied binary gate.
func (m *Machine) gate(dst, a, b *fhe.LweSample, op func(res, a, b *fhe.LweSample)) *fhe.LweSample {
	out := m.Ctx.NewCiphertext()
	op(out, a, b)
	return out
}

// Not is free (ciphertext negation flips ±1/8; no bootstrap).
func (m *Machine) Not(b *fhe.LweSample) *fhe.LweSample {
	out := m.Ctx.NewCiphertext()
	m.Ctx.GateNot(out, b)
	return out
}

// Bit-gate wrappers allocate fresh outputs; the YN/NY forms use the named
// polarity variant, and each operation preserves the fhe-bool gate contract.
func (m *Machine) Xor(a, b *fhe.LweSample) *fhe.LweSample {
	return m.gate(nil, a, b, m.Ctx.GateXor)
}
func (m *Machine) And(a, b *fhe.LweSample) *fhe.LweSample {
	return m.gate(nil, a, b, m.Ctx.GateAnd)
}
func (m *Machine) Or(a, b *fhe.LweSample) *fhe.LweSample {
	return m.gate(nil, a, b, m.Ctx.GateOr)
}

// AndNY = (not a) and b; AndYN = a and (not b); OrNY = (not a) or b.
func (m *Machine) AndNY(a, b *fhe.LweSample) *fhe.LweSample {
	return m.gate(nil, a, b, m.Ctx.GateAndNY)
}
func (m *Machine) AndYN(a, b *fhe.LweSample) *fhe.LweSample {
	return m.gate(nil, a, b, m.Ctx.GateAndYN)
}
func (m *Machine) OrNY(a, b *fhe.LweSample) *fhe.LweSample {
	return m.gate(nil, a, b, m.Ctx.GateOrNY)
}

// Mux returns trueVal when sel is 1, else falseVal (nufhe gate_mux order).
func (m *Machine) Mux(sel, trueVal, falseVal *fhe.LweSample) *fhe.LweSample {
	out := m.Ctx.NewCiphertext()
	m.Ctx.GateMux(out, sel, trueVal, falseVal)
	return out
}

// ---------------------------------------------------------------------------
// word-level operations
// ---------------------------------------------------------------------------

// WordNot is free (per-bit ciphertext negation).
func (m *Machine) WordNot(a *Word) *Word {
	out := &Word{Bits: make([]*fhe.LweSample, a.W())}
	for i := range a.Bits {
		out.Bits[i] = m.Not(a.Bits[i])
	}
	return out
}

// WordXor, WordAnd, WordOr, and wordBin apply a bitwise gate at each index;
// wordBin requires equal widths and preserves the input little-endian layout.
func (m *Machine) WordXor(a, b *Word) *Word { return m.wordBin(a, b, m.Xor) }
func (m *Machine) WordAnd(a, b *Word) *Word { return m.wordBin(a, b, m.And) }
func (m *Machine) WordOr(a, b *Word) *Word  { return m.wordBin(a, b, m.Or) }

func (m *Machine) wordBin(a, b *Word, op func(x, y *fhe.LweSample) *fhe.LweSample) *Word {
	if a.W() != b.W() {
		panic(fmt.Sprintf("vm: word width mismatch %d vs %d", a.W(), b.W()))
	}
	out := &Word{Bits: make([]*fhe.LweSample, a.W())}
	for i := range a.Bits {
		out.Bits[i] = op(a.Bits[i], b.Bits[i])
	}
	return out
}

// WordMux selects per bit: sel ? a : b (all same width).
func (m *Machine) WordMux(sel *fhe.LweSample, a, b *Word) *Word {
	if a.W() != b.W() {
		panic("vm: WordMux width mismatch")
	}
	out := &Word{Bits: make([]*fhe.LweSample, a.W())}
	for i := range a.Bits {
		out.Bits[i] = m.Mux(sel, a.Bits[i], b.Bits[i])
	}
	return out
}

// Add returns a + b mod 2^w (ripple-carry full adder; 4 gates per bit).
func (m *Machine) Add(a, b *Word) *Word {
	n := a.W()
	out := &Word{Bits: make([]*fhe.LweSample, n)}
	carry := m.zeroBit
	for i := 0; i < n; i++ {
		t := m.Xor(a.Bits[i], b.Bits[i])
		out.Bits[i] = m.Xor(t, carry)
		// carry' = (a & b) | (carry & t)
		carry = m.Or(m.And(a.Bits[i], b.Bits[i]), m.And(carry, t))
	}
	return out
}

// AddCin returns a + b + cin mod 2^w.
func (m *Machine) AddCin(a, b *Word, cin *fhe.LweSample) *Word {
	n := a.W()
	out := &Word{Bits: make([]*fhe.LweSample, n)}
	carry := cin
	for i := 0; i < n; i++ {
		t := m.Xor(a.Bits[i], b.Bits[i])
		out.Bits[i] = m.Xor(t, carry)
		carry = m.Or(m.And(a.Bits[i], b.Bits[i]), m.And(carry, t))
	}
	return out
}

// CarryOut returns the carry out of a + b (the w+1-th bit).
func (m *Machine) CarryOut(a, b *Word) *fhe.LweSample {
	n := a.W()
	carry := m.zeroBit
	for i := 0; i < n; i++ {
		t := m.Xor(a.Bits[i], b.Bits[i])
		carry = m.Or(m.And(a.Bits[i], b.Bits[i]), m.And(carry, t))
	}
	return carry
}

// Sub returns a - b mod 2^w (two's complement: a + ~b + 1).
func (m *Machine) Sub(a, b *Word) *Word {
	nb := m.WordNot(b)
	return m.AddCin(a, nb, m.oneBit)
}

// BorrowOut returns the unsigned borrow of a - b (1 iff a < b unsigned).
func (m *Machine) BorrowOut(a, b *Word) *fhe.LweSample {
	// borrow chain of the full subtractor: br' = (~a & b) | (~a & br) | (b & br)
	// = ~a & (b | br) | (b & br)
	n := a.W()
	br := m.zeroBit
	for i := 0; i < n; i++ {
		na := m.Not(a.Bits[i])
		br = m.Or(m.And(na, m.Or(b.Bits[i], br)), m.And(b.Bits[i], br))
	}
	return br
}

// LtU returns 1 iff a < b (unsigned).
func (m *Machine) LtU(a, b *Word) *fhe.LweSample { return m.BorrowOut(a, b) }

// LtS returns 1 iff a < b (signed two's complement): flip both sign bits
// (free) and compare unsigned.
func (m *Machine) LtS(a, b *Word) *fhe.LweSample {
	fa, fb := m.Clone(a), m.Clone(b)
	top := a.W() - 1
	fa.Bits[top] = m.Not(fa.Bits[top])
	fb.Bits[top] = m.Not(fb.Bits[top])
	return m.BorrowOut(fa, fb)
}

// Eq returns 1 iff a == b (per-bit xnor + AND tree).
func (m *Machine) Eq(a, b *Word) *fhe.LweSample {
	n := a.W()
	x := m.XnorBit(a.Bits[0], b.Bits[0])
	for i := 1; i < n; i++ {
		x = m.And(x, m.XnorBit(a.Bits[i], b.Bits[i]))
	}
	return x
}

// XnorBit returns a fresh ciphertext containing the XNOR of two input bits.
func (m *Machine) XnorBit(a, b *fhe.LweSample) *fhe.LweSample {
	out := m.Ctx.NewCiphertext()
	m.Ctx.GateXnor(out, a, b)
	return out
}

// ShiftLeft returns a << s for a plaintext-free encrypted shift amount
// word sh (any width; only the low log2(w) bits matter, higher shamt
// shifts fully out). Barrel shifter: for each shamt bit k, conditionally
// shift by 2^k.
func (m *Machine) ShiftLeft(a *Word, sh *Word) *Word {
	return m.barrelShift(a, sh, false, false)
}

// ShiftRightLogical: a >> sh, zero-filled.
func (m *Machine) ShiftRightLogical(a *Word, sh *Word) *Word {
	return m.barrelShift(a, sh, true, false)
}

// ShiftRightArithmetic: a >> sh, sign-filled (bit w-1).
func (m *Machine) ShiftRightArithmetic(a *Word, sh *Word) *Word {
	return m.barrelShift(a, sh, true, true)
}

// barrelShift conditionally applies power-of-two shifts selected by low shamt bits.
func (m *Machine) barrelShift(a *Word, sh *Word, right, arith bool) *Word {
	n := a.W()
	cur := m.Clone(a)
	sign := cur.Bits[n-1]
	for k := 0; (1<<uint(k)) < n && k < sh.W(); k++ {
		sel := sh.Bits[k]
		shift := 1 << uint(k)
		next := &Word{Bits: make([]*fhe.LweSample, n)}
		for i := 0; i < n; i++ {
			var src *fhe.LweSample
			if right {
				if i+shift < n {
					src = cur.Bits[i+shift]
				} else if arith {
					src = sign
				} else {
					src = m.zeroBit
				}
			} else {
				if i-shift >= 0 {
					src = cur.Bits[i-shift]
				} else {
					src = m.zeroBit
				}
			}
			next.Bits[i] = m.Mux(sel, src, cur.Bits[i])
		}
		cur = next
	}
	return cur
}

// ZeroWord returns a width-w word whose bits all reference the cached encrypted zero.
func (m *Machine) ZeroWord(w int) *Word {
	out := &Word{Bits: make([]*fhe.LweSample, w)}
	for i := range out.Bits {
		out.Bits[i] = m.zeroBit
	}
	return out
}
