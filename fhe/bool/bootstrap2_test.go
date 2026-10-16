// This test predicts the output phase for full bootstraps with exactly one
// nonzero LWE mask coefficient, including positive and negative torus values.
// It pins modulus-switch rounding, the signed rotation index, test-vector lookup,
// extraction, and keyswitch as one path without the ambiguity of random masks.
package fhebool

import (
	"math/rand"
	"testing"

	"github.com/flashnext/frost-vm/fhe/fft"
)

// Full bootstrap (with keyswitch) on hand-built samples with exactly one
// nonzero mask coefficient. For x = (a at index i, b=0):
//
//	barb = 0, bara_i = modSwitch(a)
//	p = (barb - bara_i*s_i) mod 2N
//	output phase = v[p] = mu if p < N else -mu
func TestBootstrapSingleMaskCoeff(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(31))
	proc, _ := fft.NewProcessor(p.N)
	key := NewLweKey(rnd, p)
	tlweKey := newBinaryKey(rnd, p.N)
	ks := p.MakeKeySwitchKey(rnd, key, tlweKey)
	bk := p.MakeBootstrappingKey(rnd, proc, key, tlweKey)

	twoN := 2 * p.N
	cases := []struct {
		i int
		a int32
	}{
		{0, 1}, {0, 1 << 20}, {0, -1 << 20}, {0, int32(^uint32(0) >> 1)}, // max
		{5, 12345}, {5, -12345}, {31, 1 << 30}, {63, 7},
	}
	for _, c := range cases {
		x := NewLweSample(p.LweN)
		x.A[c.i] = c.a
		res := NewLweSample(p.LweN)
		p.bootstrap(res, mu1_8, x, bk, ks, proc, nil)
		phase := lwePhase(res, key)

		bara := int(modSwitchFromTorus32(c.a, twoN))
		si := int(key.Key[c.i])
		pIdx := ((0-bara*si)%twoN + twoN) % twoN
		want := mu1_8
		if pIdx >= p.N {
			want = -mu1_8
		}
		errPhase := phase - want
		if errPhase > mu1_8/2 || errPhase < -mu1_8/2 {
			t.Fatalf("bootstrap single-mask i=%d a=%d s_i=%d bara=%d p=%d: phase=%d want=%d",
				c.i, c.a, si, bara, pIdx, phase, want)
		}
		t.Logf("i=%d a=%d s_i=%d bara=%d p=%d -> phase=%d OK", c.i, c.a, si, bara, pIdx, phase)
	}
}
