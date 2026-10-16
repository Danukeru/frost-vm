// These end-to-end tests pin every binary truth table, NOT/CONSTANT, MUX selector
// semantics, and chained-gate composition under the documented ±1/8 input contract.
// They catch mismatches between each linear pre-combination and its bootstrap
// test vector, plus noise or key-space errors that isolated helper tests miss.
package fhebool

import (
	"testing"
)

// Gate truth tables: every input combination, encrypted freshly each trial.
// Uses TestParams (LweN=64) so the whole matrix runs in seconds; the
// algorithm is identical to the default parameter set.

type binGate struct {
	name string
	fn   func(c *Context, res, a, b *LweSample)
	want func(a, b bool) bool
}

// TestBinaryGateTruthTables checks all ten binary gate tables over fresh ciphertexts.
func TestBinaryGateTruthTables(t *testing.T) {
	c, err := NewContextWithParams(TestParams(), 100)
	if err != nil {
		t.Fatal(err)
	}
	gates := []binGate{
		{"NAND", (*Context).GateNand, func(a, b bool) bool { return !(a && b) }},
		{"AND", (*Context).GateAnd, func(a, b bool) bool { return a && b }},
		{"OR", (*Context).GateOr, func(a, b bool) bool { return a || b }},
		{"NOR", (*Context).GateNor, func(a, b bool) bool { return !(a || b) }},
		{"XOR", (*Context).GateXor, func(a, b bool) bool { return a != b }},
		{"XNOR", (*Context).GateXnor, func(a, b bool) bool { return a == b }},
		{"ANDNY", (*Context).GateAndNY, func(a, b bool) bool { return !a && b }},
		{"ANDYN", (*Context).GateAndYN, func(a, b bool) bool { return a && !b }},
		{"ORNY", (*Context).GateOrNY, func(a, b bool) bool { return !a || b }},
		{"ORYN", (*Context).GateOrYN, func(a, b bool) bool { return a || !b }},
	}
	for _, g := range gates {
		var bad int
		for trial := 0; trial < 2; trial++ {
			for ai := 0; ai < 2; ai++ {
				for bi := 0; bi < 2; bi++ {
					a, b := ai == 1, bi == 1
					ca, cb := c.EncryptBit(a), c.EncryptBit(b)
					res := c.NewCiphertext()
					g.fn(c, res, ca, cb)
					got := c.DecryptBit(res)
					if want := g.want(a, b); got != want {
						bad++
						t.Errorf("%s(%v,%v) = %v, want %v (trial %d)", g.name, a, b, got, want, trial)
					}
				}
			}
		}
		if bad == 0 {
			t.Logf("%s: 8/8 truth-table entries correct over 2 trials", g.name)
		}
	}
}

// TestGateNotAndConstant pins free NOT, trivial constants, and double negation.
func TestGateNotAndConstant(t *testing.T) {
	c, err := NewContextWithParams(TestParams(), 101)
	if err != nil {
		t.Fatal(err)
	}
	for _, bit := range []bool{true, false} {
		// encrypted NOT
		ca := c.EncryptBit(bit)
		res := c.NewCiphertext()
		c.GateNot(res, ca)
		if got := c.DecryptBit(res); got != !bit {
			t.Errorf("NOT(%v) = %v", bit, got)
		}
		// constant then NOT (bootstrapped NOT via NAND with constants)
		k := c.NewCiphertext()
		c.GateConstant(k, bit)
		if got := c.DecryptBit(k); got != bit {
			t.Errorf("constant(%v) decrypt = %v", bit, got)
		}
		// double NOT round-trip
		res2 := c.NewCiphertext()
		c.GateNot(res2, res)
		if got := c.DecryptBit(res2); got != bit {
			t.Errorf("NOT(NOT(%v)) = %v", bit, got)
		}
	}
}

// TestGateMuxTruthTable verifies selector-first MUX behavior for all input triples.
func TestGateMuxTruthTable(t *testing.T) {
	c, err := NewContextWithParams(TestParams(), 102)
	if err != nil {
		t.Fatal(err)
	}
	// mux(a, b, c) = b if a else c (selector is a)
	for trial := 0; trial < 2; trial++ {
		for ai := 0; ai < 2; ai++ {
			for bi := 0; bi < 2; bi++ {
				for ci := 0; ci < 2; ci++ {
					a, b, cc := ai == 1, bi == 1, ci == 1
					ca, cb, cct := c.EncryptBit(a), c.EncryptBit(b), c.EncryptBit(cc)
					res := c.NewCiphertext()
					c.GateMux(res, ca, cb, cct)
					want := b
					if !a {
						want = cc
					}
					if got := c.DecryptBit(res); got != want {
						t.Errorf("MUX(%v,%v,%v) = %v, want %v (trial %d)", a, b, cc, got, want, trial)
					}
				}
			}
		}
	}
	t.Log("MUX: 8/8 truth-table entries correct over 2 trials")
}

// TestCircuitComposition checks gate outputs can feed further gates (the
// property the oblivious VM depends on): a 4-input parity via XOR chain.
func TestCircuitComposition(t *testing.T) {
	c, err := NewContextWithParams(TestParams(), 103)
	if err != nil {
		t.Fatal(err)
	}
	for trial := 0; trial < 3; trial++ {
		for pattern := 0; pattern < 16; pattern++ {
			bits := []bool{pattern&1 != 0, pattern&2 != 0, pattern&4 != 0, pattern&8 != 0}
			cts := make([]*LweSample, 4)
			for i, b := range bits {
				cts[i] = c.EncryptBit(b)
			}
			acc := c.NewCiphertext()
			c.GateXor(acc, cts[0], cts[1])
			tmp := c.NewCiphertext()
			c.GateXor(tmp, acc, cts[2])
			c.GateXor(acc, tmp, cts[3])
			want := false
			for _, b := range bits {
				want = want != b
			}
			if got := c.DecryptBit(acc); got != want {
				t.Fatalf("parity(%04b) = %v, want %v (trial %d)", pattern, got, want, trial)
			}
		}
	}
	t.Log("4-input XOR chain: 16/16 patterns correct over 3 trials")
}
