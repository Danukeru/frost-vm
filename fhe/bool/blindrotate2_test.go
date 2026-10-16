// This test pins composition of sparse, nonzero blind rotations: the final TLWE
// phase must equal X^(sum(a_i*s_i) mod 2N) applied to the initial test vector.
// It protects accumulation order, encrypted key-bit selection, and rotation signs
// across multiple external products, while skipping zero rotations as the driver does.
package fhebool

import (
	"math/rand"
	"testing"

	"github.com/flashnext/frost-vm/fhe/fft"
)

// Multi-key blind rotation with a sparse bara vector: the phase polynomial
// must be X^{sum(a_i s_i) mod 2N} * tvb.
func TestBlindRotateMultiKeySparse(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(23))
	proc, _ := fft.NewProcessor(p.N)
	lweKey := NewLweKey(rnd, p)
	tlweKey := newBinaryKey(rnd, p.N)
	bk := p.MakeBootstrappingKey(rnd, proc, lweKey, tlweKey)
	sc := p.newScratch()

	mu := mu1_8
	tv := make([]int32, p.N)
	for i := range tv {
		tv[i] = mu
	}
	tvb := make([]int32, p.N)
	mulByXai(tvb, 123, tv) // arbitrary barb

	// sparse bara: a few nonzero rotations
	entries := []struct {
		i int
		a int32
	}{
		{2, 100}, {5, 900}, {11, 1777}, {17, 3}, {23, 2000},
	}

	acc := p.newTlweSample()
	p.noiselessTrivial(acc, tvb)
	cur, other := acc, sc.tmp
	bara := make([]int32, len(lweKey.Key))
	for _, e := range entries {
		bara[e.i] = e.a
	}
	for i := range bara {
		if bara[i] == 0 {
			continue
		}
		for m := 0; m < 2; m++ {
			mulByXaiMinusOne(other.poly(m), int(bara[i]), cur.poly(m))
		}
		p.externMulTo(other, bk.rows[i], proc, sc)
		p.tlweAddTo(other, cur)
		cur, other = other, cur
	}

	// expected: X^{sum(a_i s_i)} applied on top of tvb
	exp := 0
	for _, e := range entries {
		exp += int(e.a) * int(lweKey.Key[e.i])
	}
	exp %= 2 * p.N
	want := make([]int32, p.N)
	mulByXai(want, exp, tvb)

	phase := p.tlwePhasePoly(cur, tlweKey)
	var maxErr int32
	for k := 0; k < p.N; k++ {
		d := phase[k] - want[k]
		if d > maxErr {
			maxErr = d
		}
		if d < -maxErr {
			maxErr = -d
		}
	}
	t.Logf("multi-key blind rotate: exponent=%d maxErr=%d LSB", exp, maxErr)
	// external-product noise over len(entries) steps: ~sqrt(steps)*300k LSB
	if maxErr > 30_000_000 {
		t.Fatalf("multi-key blind rotate wrong: %d LSB", maxErr)
	}
}
