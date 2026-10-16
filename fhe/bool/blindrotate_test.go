// These tests pin a single encrypted-key blind-rotation step and coefficient-zero
// extraction. They compare the TLWE phase polynomial against the expected
// negacyclic rotation and require extraction to preserve P[0] exactly. These are
// decisive sign/index checks for the mux-rotate formula and extraction convention;
// a trivial-mask bootstrap alone would not exercise the same invariants.
package fhebool

import (
	"math/rand"
	"testing"

	"github.com/flashnext/frost-vm/fhe/fft"
)

// Single-key blind rotation: start from acc = (0, tvb) with tvb = X^{2N-barb}*v,
// apply one mux_rotate step for key i with rotation a. The phase polynomial
// under the TLWE key must equal X^{a*s_i} * tvb (i.e. tvb if s_i=0, X^a*tvb
// if s_i=1), up to the external-product noise.
func TestBlindRotateSingleKey(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(21))
	proc, _ := fft.NewProcessor(p.N)
	lweKey := NewLweKey(rnd, p)
	tlweKey := newBinaryKey(rnd, p.N)
	bk := p.MakeBootstrappingKey(rnd, proc, lweKey, tlweKey)
	sc := p.newScratch()

	// test vector: step function encoded as [mu]*N
	mu := mu1_8
	tv := make([]int32, p.N)
	for i := range tv {
		tv[i] = mu
	}

	for _, i := range []int{3, 10} {
		for _, a := range []int{0, 1, 777, 1500} {
			if a == 0 {
				continue // skipped by the loop; nothing to test
			}
			acc := p.newTlweSample()
			tvb := make([]int32, p.N)
			mulByXai(tvb, (2*p.N-a/2)%(2*p.N), tv) // arbitrary barb = a/2
			p.noiselessTrivial(acc, tvb)

			// one mux_rotate step (same code as blindRotateAndExtract)
			other := sc.tmp
			for m := 0; m < 2; m++ {
				mulByXaiMinusOne(other.poly(m), a, acc.poly(m))
			}
			p.externMulTo(other, bk.rows[i], proc, sc)
			p.tlweAddTo(other, acc)

			// phase polynomial must be X^{a*s_i} * tvb
			si := lweKey.Key[i]
			want := make([]int32, p.N)
			if si == 1 {
				mulByXai(want, a, tvb)
			} else {
				copy(want, tvb)
			}
			phase := p.tlwePhasePoly(other, tlweKey)
			var maxErr int32
			for k := 0; k < p.N; k++ {
				e := phase[k] - want[k]
				if e > maxErr {
					maxErr = e
				}
				if e < -maxErr {
					maxErr = -e
				}
			}
			t.Logf("i=%d s_i=%d a=%d: max phase err = %d LSB", i, si, a, maxErr)
			if maxErr > 3_000_000 {
				t.Fatalf("single-key blind rotate wrong (i=%d a=%d): %d LSB", i, a, maxErr)
			}
		}
	}
}

// Extraction convention: for acc = (0, P) noiseless, the extracted LWE
// sample's phase under the extracted key (tlweKey as a vector) must equal
// P[0] exactly.
func TestExtractConvention(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(22))
	tlweKey := newBinaryKey(rnd, p.N)
	P := make([]int32, p.N)
	for i := range P {
		P[i] = int32(rnd.Uint64())
	}
	acc := p.newTlweSample()
	p.noiselessTrivial(acc, P)
	res := NewLweSample(p.N)
	p.extractLwe(res, acc)
	exKey := &LweKey{Key: tlweKey}
	got := lwePhase(res, exKey)
	if got != P[0] {
		t.Fatalf("extract phase = %d, want P[0] = %d", got, P[0])
	}
}
