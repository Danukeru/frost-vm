// Package vm tests RV64 load-width extension through an oblivious machine program.
//
// TestLoadExtWidths pins the width-aware load extension bug exposed by the
// 20-value Fibonacci run: a 32-bit load must preserve bit 8 of value 377.
// It also checks that LB sign-extends 0xff while stores preserve the loaded result.
// Only bits at or above the access width may be replaced by sign/zero fill.
// The machine executes all candidate load widths; assertions decrypt RAM afterward.
package vm

import (
	"testing"
)

// TestLoadExtWidths pins the load-extension semantics that the full
// fibonacci run exposed as a bug: LW of a value with bit 8 set (377)
// must keep bits 8..31 as real data; only bits at or above the access
// width get the sign/zero fill. Before the fix, extension replaced
// everything >= bit 8 for every width, so LW 377 came back as 121.
//
//	0:  LUI a0, 0x1          ; a0 = 0x1000 (data base)
//	4:  ADDI t0, zero, 377
//	8:  SW t0, 56(a0)        ; mem[56..59] = 377
//	12: LW t1, 56(a0)        ; sign-extended load
//	16: SW t1, 60(a0)        ; mem[60..63] = loaded value
//	20: ADDI t2, zero, 0xFF
//	24: SB t2, 70(a0)        ; mem[70] = 0xFF
//	28: LB a1, 70(a0)        ; sign-extends to all ones
//	32: SD a1, 72(a0)        ; mem[72..79] = 0xFF..FF
//	36: ECALL
func TestLoadExtWidths(t *testing.T) {
	prog := []uint32{
		(uint32(1)<<12 | 10<<7) | OpLUI, // lui a0, 0x1
		encI(OpImm, 5, 0, 377, 0x0),
		encS(56, 10, 5, F3SW),
		encI(OpLoad, 6, 10, 56, F3LW),
		encS(60, 10, 6, F3SW),
		encI(OpImm, 7, 0, 0xFF, 0x0),
		encS(70, 10, 7, F3SB),
		encI(OpLoad, 11, 10, 70, F3LB),
		encS(72, 10, 11, F3SD),
		encI(OpSystem, 0, 0, 0, 0),
	}
	rv := newTestRV(t, 16, 12, 128, prog)
	rv.Run(20)

	// mem[60..63] must be 377 (bit 8 preserved through LW)
	var got uint64
	for b := 0; b < 4; b++ {
		got |= rv.DecryptMemByte(60+b) << (8 * uint(b))
	}
	if got != 377 {
		t.Errorf("LW 377 round-trip: mem[60..63] = %d, want 377 (bit 8 dropped by load extension?)", got)
	}
	// mem[72..79] must be 0xFF..FF (LB sign extension)
	for b := 0; b < 8; b++ {
		if v := rv.DecryptMemByte(72 + b); v != 0xFF {
			t.Errorf("LB 0xFF sign-extension: mem[%d] = %#x, want 0xff", 72+b, v)
		}
	}
}
