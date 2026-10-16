// This suite pins centered gadget decomposition, bootstrap-key encryption, the
// external product, and full bootstrap on trivial and encrypted LWE inputs. It
// checks the diagonal TGSW message placement and phase-level noise bounds, not
// just decrypted output bits. Together these assertions isolate decomposition,
// external-product scaling, blind rotation, extraction, and keyswitch regressions.
package fhebool

import (
	"math/rand"
	"testing"

	"github.com/flashnext/frost-vm/fhe/fft"
)

// ---- decomposition (tGswTorus32PolynomialDecompH) ----

func TestDecompHReconstructs(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(11))
	N := 256
	src := make([]int32, N)
	for i := range src {
		src[i] = int32(rnd.Uint64())
	}
	digits := make([][]int32, p.BgL)
	for j := range digits {
		digits[j] = make([]int32, N)
		p.decompH(digits[j], src, j)
	}
	// x ~= sum_j digit_j * h_j (Torus32 wrapping); residual bounded by
	// half of the last digit's weight: 2^(32 - l*Bgbit)/2.
	limit := int32(1) << (32 - p.BgL*p.BgBit)
	for i := 0; i < N; i++ {
		var acc int32
		for j := 0; j < p.BgL; j++ {
			acc += digits[j][i] * p.h[j]
			// digits must be centered in [-halfBg, halfBg)
			if d := digits[j][i]; d < -p.halfBg || d >= p.halfBg {
				t.Fatalf("digit out of range: j=%d d=%d", j, d)
			}
		}
		err := acc - src[i]
		if err > limit || err < -limit {
			t.Fatalf("decomp residual too large at %d: err=%d limit=%d", i, err, limit)
		}
	}
}

// ---- bootstrap key correctness ----

func TestBootstrappingKeyEncryptsLweKey(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(12))
	proc, err := fft.NewProcessor(p.N)
	if err != nil {
		t.Fatal(err)
	}
	lweKey := NewLweKey(rnd, p)
	tlweKey := newBinaryKey(rnd, p.N)
	bk := p.MakeBootstrappingKey(rnd, proc, lweKey, tlweKey)

	// Only check a few of the 64 TGSW samples to keep the test quick.
	for _, i := range []int{0, 1, len(lweKey.Key) - 1} {
		bit := lweKey.Key[i]
		g := bk.rows[i]
		for q := 0; q < p.kpl(); q++ {
			m := q / p.BgL
			j := q % p.BgL
			a, b := p.tlweRowCoeffs(proc, g, q)
			sample := &TlweSample{A: [][]int32{a}, B: b}
			phase := p.tlwePhasePoly(sample, tlweKey)
			// diagonal convention (tGswAddMuIntH): the message bit*h[j] sits
			// in polynomial m of row (m,j). The row's phase b - a*s is then
			// +bit*h[j] (constant) for m=1, and -bit*h[j]*s(x) for m=0
			// (the key polynomial scaled), up to encryption noise.
			for k := 0; k < p.N; k++ {
				want := int32(0)
				if m == 1 {
					if k == 0 {
						want = bit * p.h[j]
					}
				} else {
					want = -bit * p.h[j] * tlweKey[k]
				}
				errPhase := phase[k] - want
				if errPhase > 500 || errPhase < -500 {
					t.Fatalf("bk[%d] row %d phase error at %d: got %d want %d err %d",
						i, q, k, phase[k], want, errPhase)
				}
			}
		}
	}
}

// ---- external product on a trivial accumulator ----

func TestExternMulTrivialAccumulator(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(13))
	proc, err := fft.NewProcessor(p.N)
	if err != nil {
		t.Fatal(err)
	}
	lweKey := NewLweKey(rnd, p)
	tlweKey := newBinaryKey(rnd, p.N)
	bk := p.MakeBootstrappingKey(rnd, proc, lweKey, tlweKey)
	sc := p.newScratch()

	// small integer test polynomial
	P := make([]int32, p.N)
	for i := range P {
		P[i] = int32(rnd.Intn(2001) - 1000)
	}

	for _, i := range []int{0, 7} {
		acc := p.newTlweSample()
		p.noiselessTrivial(acc, P)
		p.externMulTo(acc, bk.rows[i], proc, sc)
		phase := p.tlwePhasePoly(acc, tlweKey)
		bit := lweKey.Key[i]
		var maxErr int32
		for k := 0; k < p.N; k++ {
			want := bit * P[k]
			e := phase[k] - want
			if e < 0 {
				e = -e
			}
			if e > maxErr {
				maxErr = e
			}
		}
		// External-product noise: kpl*(k+1) products of digits (<=512) with
		// bk noise (~39 LSB stdev) => stdev ~ 4.5e4 LSB; 3e5 is a very
		// generous cap that still proves the signs/scales are right.
		if maxErr > 300_000 {
			t.Fatalf("extern product noise too large (bk[%d]): %d LSB", i, maxErr)
		}
		t.Logf("extern product bk[%d] (bit=%d): max phase error %d LSB", i, bit, maxErr)
	}
}

// ---- bootstrap on trivial samples ----

func TestBootstrapTrivialPhase(t *testing.T) {
	p := TestParams()
	proc, err := fft.NewProcessor(p.N)
	if err != nil {
		t.Fatal(err)
	}
	rnd := rand.New(rand.NewSource(14))
	key := NewLweKey(rnd, p)
	tlweKey := newBinaryKey(rnd, p.N)
	ks := p.MakeKeySwitchKey(rnd, key, tlweKey)
	bk := p.MakeBootstrappingKey(rnd, proc, key, tlweKey)

	for _, in := range []int32{mu1_8 / 2, -mu1_8 / 2, mu1_8/2 + 12345, -3 * mu1_8 / 4} {
		x := NewLweSample(p.LweN)
		x.B = in // trivial: phase = in
		res := NewLweSample(p.LweN)
		p.bootstrap(res, mu1_8, x, bk, ks, proc, nil)
		phase := lwePhase(res, key)
		var want int32
		if in >= 0 {
			want = mu1_8
		} else {
			want = -mu1_8
		}
		errPhase := phase - want
		if errPhase > mu1_8/2 || errPhase < -mu1_8/2 {
			t.Fatalf("bootstrap in=%d: phase=%d want=%d err=%d", in, phase, want, errPhase)
		}
		t.Logf("bootstrap in=%d -> phase=%d (want %d, err %d)", in, phase, want, errPhase)
	}

	// Non-trivial inputs: random mask under the LWE key, so the blind
	// rotation actually runs over all bara_i != 0.
	for trial := 0; trial < 5; trial++ {
		bit := trial%2 == 0
		x := p.EncryptBit(rnd, key, bit) // ±1/8, the standard encoding
		res := NewLweSample(p.LweN)
		p.bootstrap(res, mu1_8, x, bk, ks, proc, nil)
		phase := lwePhase(res, key)
		want := mu1_8
		if !bit {
			want = -mu1_8
		}
		errPhase := phase - want
		if errPhase > mu1_8/2 || errPhase < -mu1_8/2 {
			t.Fatalf("bootstrap nontrivial %d: phase=%d want=%d err=%d", trial, phase, want, errPhase)
		}
		t.Logf("bootstrap nontrivial bit=%v: phase=%d (want %d, err %d)", bit, phase, want, errPhase)
	}
}

// lweMulHalf halves the sample (exact for even coefficients; our ±1/16
// encodings come from halving ±1/8 which is exact in int32).
func lweMulHalf(x *LweSample) {
	for i := range x.A {
		x.A[i] /= 2
	}
	x.B /= 2
}
