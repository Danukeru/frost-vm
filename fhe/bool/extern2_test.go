// This test applies a real encrypted TGSW key bit to a nontrivial random TLWE
// sample and checks that the output phase is s_i times the input phase.
// A nonzero mask exercises the full b - a*s algebra and guards the diagonal
// message convention: trivial-accumulator tests can pass even with wrong signs.
package fhebool

import (
	"math/rand"
	"testing"

	"github.com/flashnext/frost-vm/fhe/fft"
)

// External product on a NON-trivial accumulator: a proper random TLWE
// sample under the TLWE key. The result phase must be s_i * phase(x).
func TestExternMulNonTrivialAccumulator(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(41))
	proc, _ := fft.NewProcessor(p.N)
	lweKey := NewLweKey(rnd, p)
	tlweKey := newBinaryKey(rnd, p.N)
	bk := p.MakeBootstrappingKey(rnd, proc, lweKey, tlweKey)
	sc := p.newScratch()

	for _, i := range []int{1, 4} {
		// random TLWE sample: mask uniform, body = mask*s + msg + noise
		a := randUniformTorus32(rnd, p.N)
		msg := make([]int32, p.N)
		for k := range msg {
			msg[k] = int32(rnd.Intn(2001) - 1000)
		}
		b := make([]int32, p.N)
		copy(b, msg)
		for u, av := range a {
			for v, sv := range tlweKey {
				if sv == 0 {
					continue
				}
				k := u + v
				if k >= p.N {
					k -= p.N
					b[k] -= av
					continue
				}
				b[k] += av
			}
		}
		for k := range b {
			b[k] += randGaussianTorus32(rnd, 1, p.BsStdev)[0]
		}
		acc := &TlweSample{A: [][]int32{a}, B: b}

		// phase before
		phaseBefore := p.tlwePhasePoly(acc, tlweKey)

		p.externMulTo(acc, bk.rows[i], proc, sc)
		phaseAfter := p.tlwePhasePoly(acc, tlweKey)

		si := lweKey.Key[i]
		var maxErr int32
		for k := 0; k < p.N; k++ {
			want := si * phaseBefore[k]
			d := phaseAfter[k] - want
			if d > maxErr {
				maxErr = d
			}
			if d < -maxErr {
				maxErr = -d
			}
		}
		t.Logf("extern mul on random TLWE: bk[%d] s_i=%d maxErr=%d LSB", i, si, maxErr)
		if maxErr > 5_000_000 {
			t.Fatalf("external product wrong on non-trivial accumulator: %d LSB", maxErr)
		}
	}
}
