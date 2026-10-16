// Package vm tests encrypted word arithmetic, comparisons, shifts, and selection.
//
// These tests compare decrypted gate results with plaintext RV64 expectations.
// Eight-bit cases pin wraparound, carry/borrow, signedness, logic, and mux behavior.
// Shift cases verify that only log2(width) low shift-amount bits are consulted.
// Sixty-four-bit cases cover real-width arithmetic and sign-boundary behavior.
// The tests target result semantics; machine-level decode and halt are tested elsewhere.
package vm

import (
	"testing"

	fhe "github.com/flashnext/frost-vm/fhe/bool"
)

// one shared machine (key generation dominates)
var shared *Machine

// testMachine reuses one test FHE context so key generation is shared across tests.
func testMachine(t *testing.T) *Machine {
	t.Helper()
	if shared == nil {
		ctx, err := fhe.NewContextWithParams(fhe.TestParams(), 4242)
		if err != nil {
			t.Fatalf("context: %v", err)
		}
		shared = NewMachine(ctx)
	}
	return shared
}

// uMask truncates v to the low w bits, matching modulo-2^w word arithmetic.
func uMask(v uint64, w int) uint64 {
	if w >= 64 {
		return v
	}
	return v & ((uint64(1) << uint(w)) - 1)
}

// sign-extend a w-bit value to int64
func sExt(v uint64, w int) int64 {
	if w >= 64 {
		return int64(v)
	}
	sign := int64(1) << uint(w-1)
	return (int64(v) ^ sign) - sign
}

// TestWordAddSub8 pins eight-bit addition and subtraction modulo 2^8.
func TestWordAddSub8(t *testing.T) {
	m := testMachine(t)
	cases := []struct{ a, b uint64 }{
		{0, 0}, {0, 255}, {1, 255}, {255, 1}, {127, 128}, {200, 55},
		{17, 238}, {85, 85}, {1, 1}, {128, 128},
	}
	for _, c := range cases {
		a := m.EncryptWord(c.a, 8)
		b := m.EncryptWord(c.b, 8)
		got := m.DecryptWord(m.Add(a, b))
		want := uMask(c.a+c.b, 8)
		if got != want {
			t.Errorf("add8(%d,%d) = %d, want %d", c.a, c.b, got, want)
		}
		got = m.DecryptWord(m.Sub(a, b))
		want = uMask(c.a-c.b, 8)
		if got != want {
			t.Errorf("sub8(%d,%d) = %d, want %d", c.a, c.b, got, want)
		}
	}
}

// TestWordCarryBorrow8 checks carry-out, unsigned/signed less-than, and equality.
func TestWordCarryBorrow8(t *testing.T) {
	m := testMachine(t)
	cases := []struct{ a, b uint64 }{
		{0, 0}, {255, 0}, {255, 1}, {0, 1}, {127, 128}, {128, 127}, {200, 55},
	}
	for _, c := range cases {
		a := m.EncryptWord(c.a, 8)
		b := m.EncryptWord(c.b, 8)
		if got := m.Ctx.DecryptBit(m.CarryOut(a, b)); got != (c.a+c.b > 255) {
			t.Errorf("carry(%d,%d) = %v", c.a, c.b, got)
		}
		if got := m.Ctx.DecryptBit(m.LtU(a, b)); got != (c.a < c.b) {
			t.Errorf("ltu(%d,%d) = %v", c.a, c.b, got)
		}
		sa, sb := sExt(c.a, 8), sExt(c.b, 8)
		if got := m.Ctx.DecryptBit(m.LtS(a, b)); got != (sa < sb) {
			t.Errorf("lts(%d,%d) [signed %d<%d] = %v", c.a, c.b, sa, sb, got)
		}
		if got := m.Ctx.DecryptBit(m.Eq(a, b)); got != (c.a == c.b) {
			t.Errorf("eq(%d,%d) = %v", c.a, c.b, got)
		}
	}
}

// TestWordShifts8 pins eight-bit logical/arithmetic shifts and shift-width masking.
func TestWordShifts8(t *testing.T) {
	m := testMachine(t)
	// The barrel shifter only consults shamt bits below log2(width) — the
	// same masking RV64 applies (shamt is 6 bits for 64-bit words). At
	// width 8 a shamt of 8 therefore means shift-by-0; that case pins it.
	cases := []struct {
		a, sh uint64
	}{
		{0x81, 0}, {0x81, 1}, {0x81, 3}, {0x81, 7}, {0x81, 8}, {0x0F, 4},
	}
	for _, c := range cases {
		a := m.EncryptWord(c.a, 8)
		sh := m.EncryptWord(c.sh, 4)
		s := c.sh % 8 // shamt masked to 3 bits at width 8
		wantL := uMask(c.a<<s, 8)
		if got := m.DecryptWord(m.ShiftLeft(a, sh)); got != wantL {
			t.Errorf("sll8(%#x,%d) = %#x, want %#x", c.a, c.sh, got, wantL)
		}
		wantR := c.a >> s
		if got := m.DecryptWord(m.ShiftRightLogical(a, sh)); got != uMask(wantR, 8) {
			t.Errorf("srl8(%#x,%d) = %#x, want %#x", c.a, c.sh, got, uMask(wantR, 8))
		}
		wantA := uint64(sExt(c.a, 8) >> int64(s))
		if got := m.DecryptWord(m.ShiftRightArithmetic(a, sh)); got != uMask(wantA, 8) {
			t.Errorf("sra8(%#x,%d) = %#x, want %#x", c.a, c.sh, got, uMask(wantA, 8))
		}
	}
}

// TestWordMuxAndLogic checks word XOR/AND/OR plus the mux selector polarity.
func TestWordMuxAndLogic(t *testing.T) {
	m := testMachine(t)
	xorCases := []struct{ a, b uint64 }{{0xAA, 0x55}, {0xFF, 0xFF}, {0x0F, 0xF0}, {0, 0x9A}}
	for _, c := range xorCases {
		a := m.EncryptWord(c.a, 8)
		b := m.EncryptWord(c.b, 8)
		if got := m.DecryptWord(m.WordXor(a, b)); got != uMask(c.a^c.b, 8) {
			t.Errorf("xor(%#x,%#x) = %#x", c.a, c.b, got)
		}
		if got := m.DecryptWord(m.WordAnd(a, b)); got != uMask(c.a&c.b, 8) {
			t.Errorf("and(%#x,%#x) = %#x", c.a, c.b, got)
		}
		if got := m.DecryptWord(m.WordOr(a, b)); got != uMask(c.a|c.b, 8) {
			t.Errorf("or(%#x,%#x) = %#x", c.a, c.b, got)
		}
	}
	// mux: sel=1 -> a, sel=0 -> b
	for _, sel := range []bool{true, false} {
		s := m.Ctx.EncryptBit(sel)
		a := m.EncryptWord(0x12, 8)
		b := m.EncryptWord(0x34, 8)
		want := uint64(0x34)
		if sel {
			want = 0x12
		}
		if got := m.DecryptWord(m.WordMux(s, a, b)); got != want {
			t.Errorf("mux(sel=%v) = %#x, want %#x", sel, got, want)
		}
	}
}

// TestWord64Ops pins 64-bit wraparound arithmetic and signed/unsigned boundaries.
func TestWord64Ops(t *testing.T) {
	m := testMachine(t)
	// a few 64-bit cases: wraparound and sign boundaries
	addCases := []struct{ a, b uint64 }{
		{0xFFFFFFFFFFFFFFFF, 1},
		{0x7FFFFFFFFFFFFFFF, 1},
		{0xDEADBEEFCAFEBABE, 0x1234567890ABCDEF},
	}
	for _, c := range addCases {
		a := m.EncryptWord(c.a, 64)
		b := m.EncryptWord(c.b, 64)
		if got := m.DecryptWord(m.Add(a, b)); got != c.a+c.b {
			t.Errorf("add64(%#x,%#x) = %#x, want %#x", c.a, c.b, got, c.a+c.b)
		}
		if got := m.DecryptWord(m.Sub(a, b)); got != c.a-c.b {
			t.Errorf("sub64(%#x,%#x) = %#x, want %#x", c.a, c.b, got, c.a-c.b)
		}
	}
	// signed compare at INT64 boundaries
	sa := m.EncryptWord(0x8000000000000000, 64) // INT64_MIN
	sb := m.EncryptWord(0x7FFFFFFFFFFFFFFF, 64) // INT64_MAX
	if !m.Ctx.DecryptBit(m.LtS(sa, sb)) {
		t.Errorf("lts64(INT64_MIN, INT64_MAX) = false")
	}
	if m.Ctx.DecryptBit(m.LtS(sb, sa)) {
		t.Errorf("lts64(INT64_MAX, INT64_MIN) = true")
	}
	if m.Ctx.DecryptBit(m.LtU(sa, sb)) {
		t.Errorf("ltu64(2^63, 2^63-1) = true, want false")
	}
}

// TestWord64Shifts checks RV64-width left, logical-right, and sign-filled shifts.
func TestWord64Shifts(t *testing.T) {
	m := testMachine(t)
	// one case per shift kind, exercising shamt bits 0..5
	plain := uint64(0x0123456789ABCDEF)
	a := m.EncryptWord(plain, 64)
	sh := m.EncryptWord(13, 6)
	if got := m.DecryptWord(m.ShiftLeft(a, sh)); got != plain<<13 {
		t.Errorf("sll64 = %#x, want %#x", got, plain<<13)
	}
	if got := m.DecryptWord(m.ShiftRightLogical(a, sh)); got != plain>>13 {
		t.Errorf("srl64 = %#x", got)
	}
	nv := int64(-255)
	wantA := uint64(nv >> 4) // arithmetic shift keeps sign (computed, not constant)
	neg := m.EncryptWord(0xFFFFFFFFFFFFFF01, 64) // -255
	sh2 := m.EncryptWord(4, 6)
	if got := m.DecryptWord(m.ShiftRightArithmetic(neg, sh2)); got != wantA {
		t.Errorf("sra64 = %#x, want %#x", got, wantA)
	}
	// shamt >= width: RV64 masks shamt to 6 bits, so 64 is legal and means
	// shift by 0? No: RV64 SLL uses shamt mod 64. Our barrel shifts by the
	// raw shamt value; document behavior for shamt>=64 via masking upstream.
	// Here verify shamt=63 fully drains the word.
	sh3 := m.EncryptWord(63, 6)
	if got := m.DecryptWord(m.ShiftRightLogical(a, sh3)); got != 0 {
		t.Errorf("srl64 by 63 = %#x, want 0", got)
	}
}
