// This test constructs a noiseless, diagonal TGSW and a trivial TLWE accumulator
// so the external product must equal the exact decompH digit reconstruction.
// The zero-error assertion pins row placement, Fourier multiplication, and scale
// independently of encryption noise; it catches algebraic errors that a loose
// statistical noise bound or trivial-mask decrypt test could conceal.
package fhebool

import (
	"math/rand"
	"testing"

	"github.com/flashnext/frost-vm/fhe/fft"
)

// Exact-arithmetic check of the external product: with a noiseless TGSW
// (rows carry the constant h_j in polynomial m, diagonal convention) and a
// trivial accumulator (0, P), the result body must equal the exact integer
// reconstruction sum_j digit_j(P)*h_j — zero error, no noise involved.
func TestExternMulNoiselessExact(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(31))
	proc, err := fft.NewProcessor(p.N)
	if err != nil {
		t.Fatal(err)
	}
	g := p.newTgswSampleFFT()
	for q := 0; q < p.kpl(); q++ {
		m := q / p.BgL
		j := q % p.BgL
		row := make([]int32, p.N)
		row[0] = p.h[j]
		zero := make([]int32, p.N)
		if m == 0 {
			proc.PolyToFourier(g.row(q, 0), row)
			proc.PolyToFourier(g.row(q, 1), zero)
		} else {
			proc.PolyToFourier(g.row(q, 0), zero)
			proc.PolyToFourier(g.row(q, 1), row)
		}
	}

	P := make([]int32, p.N)
	for i := range P {
		P[i] = int32(rnd.Intn(2001) - 1000)
	}
	acc := p.newTlweSample()
	p.noiselessTrivial(acc, P)
	sc := p.newScratch()
	p.externMulTo(acc, g, proc, sc)

	digits := make([][]int32, p.BgL)
	for j := range digits {
		digits[j] = make([]int32, p.N)
		p.decompH(digits[j], P, j)
	}
	want := make([]int32, p.N)
	for j := 0; j < p.BgL; j++ {
		hj := p.h[j]
		for i := range want {
			want[i] += digits[j][i] * hj
		}
	}
	var maxErr int32
	for i := range want {
		e := acc.B[i] - want[i]
		if e > maxErr {
			maxErr = e
		}
		if e < -maxErr {
			maxErr = -e
		}
	}
	if maxErr != 0 {
		t.Fatalf("noiseless external product not exact: %d LSB", maxErr)
	}
}
