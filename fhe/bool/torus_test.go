// This suite pins Torus32 modulus switching, exact phase conversion, negacyclic
// shifts, LWE round trips, keyswitch message/noise bounds, and centered key noise.
// In particular, regression cases preserve nufhe semantics for arbitrary positive
// denominators and libtfhe integral-part truncation before dtot32 scaling; the
// keyset-sum assertion catches loss of whole-keyset noise recentering.
package fhebool

import (
	"math"
	"math/rand"
	"testing"
)

// ---- modSwitchFromTorus32 ----

func TestModSwitchRoundTrip(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	for _, M := range []int{2, 8, 1024, 2048} {
		interv := float64(((uint64(1) << 63) / uint64(M)) * 2)
		for i := 0; i < 2000; i++ {
			x := int32(rnd.Uint64())
			ms := modSwitchFromTorus32(x, M)
			if ms < 0 || ms >= int32(M) {
				t.Fatalf("modSwitch out of range: %d (M=%d)", ms, M)
			}
			// ms must be the nearest multiple index of x on the torus:
			// |x/2^32 - ms/M| <= 1/(2M) (mod 1).
			err := t32tod(x) - float64(uint32(ms))/float64(M)
			err = err - math.Floor(err+0.5) // wrap to [-0.5, 0.5)
			if math.Abs(err) > 1/float64(2*M)+1e-9 {
				t.Fatalf("modSwitch not nearest: x=%d M=%d ms=%d err=%v (interv=%v)", x, M, ms, err, interv)
			}
		}
	}
}

// TestPhaseToT32 pins signed rational phases, zero, and Torus32 wraparound.
func TestPhaseToT32(t *testing.T) {
	cases := []struct {
		num, denom int32
		want       int32
	}{
		{1, 8, 1 << 29},
		{-1, 8, -(1 << 29)}, // (-1 mod 8)=7 -> 7*2^29 = -2^29 mod 2^32
		{1, 4, 1 << 30},
		{0, 8, 0},
		{1, 2, math.MinInt32}, // 2^31 wraps to MinInt32 as int32
	}
	for _, c := range cases {
		got := phaseToT32(c.num, c.denom)
		if got != c.want {
			t.Fatalf("phaseToT32(%d,%d) = %d, want %d", c.num, c.denom, got, c.want)
		}
	}
}

// ---- negacyclic shifts ----

// naiveMulXai computes X^a * p mod (X^N+1) the slow way, for table-testing.
func naiveMulXai(p []int32, a int) []int32 {
	N := len(p)
	out := make([]int32, N)
	for i := 0; i < N; i++ {
		j := i + a
		v := p[i]
		if (j/N)%2 == 1 {
			v = -v
		}
		out[j%N] += v
	}
	return out
}

// TestMulByXai compares every rotation in [0,2N) with a naive negacyclic product.
func TestMulByXai(t *testing.T) {
	rnd := rand.New(rand.NewSource(2))
	N := 64
	p := make([]int32, N)
	for i := range p {
		p[i] = int32(rnd.Uint64())
	}
	for a := 0; a < 2*N; a++ {
		var got []int32
		mulByXaiBuf := make([]int32, N)
		mulByXai(mulByXaiBuf, a, p)
		got = mulByXaiBuf
		want := naiveMulXai(p, a)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("mulByXai(a=%d) mismatch at %d: got %d want %d", a, i, got[i], want[i])
			}
		}
	}
}

// TestMulByXaiMinusOne pins the rotation-minus-identity operator for [0,2N).
func TestMulByXaiMinusOne(t *testing.T) {
	rnd := rand.New(rand.NewSource(3))
	N := 64
	p := make([]int32, N)
	for i := range p {
		p[i] = int32(rnd.Uint64())
	}
	for a := 0; a < 2*N; a++ {
		got := make([]int32, N)
		mulByXaiMinusOne(got, a, p)
		shifted := naiveMulXai(p, a)
		for i := range got {
			if got[i] != shifted[i]-p[i] {
				t.Fatalf("mulByXaiMinusOne(a=%d) mismatch at %d", a, i)
			}
		}
	}
}

// ---- LWE encrypt/decrypt ----

func TestLweEncryptDecrypt(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(4))
	key := NewLweKey(rnd, p)
	for i := 0; i < 50; i++ {
		bit := rnd.Intn(2) == 0
		ct := p.EncryptBit(rnd, key, bit)
		if got := p.DecryptBit(key, ct); got != bit {
			t.Fatalf("encrypt/decrypt round trip failed: want %v got %v", bit, got)
		}
	}
}

// ---- KeySwitch ----

func TestKeySwitchPreservesMessage(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(5))
	outKey := NewLweKey(rnd, p) // size 64
	inKey := newBinaryKey(rnd, p.N)
	ks := p.MakeKeySwitchKey(rnd, outKey, inKey)

	for i := 0; i < 20; i++ {
		bit := rnd.Intn(2) == 0
		// encrypt under the input key (as an LWE sample of size N)
		inSample := p.encryptMu(rnd, &LweKey{Key: inKey}, boolToT32(bit), p.KsStdev)
		out := NewLweSample(p.LweN)
		p.KeySwitch(out, ks, inSample)
		if got := p.DecryptBit(outKey, out); got != bit {
			t.Fatalf("keyswitch changed message: want %v got %v (trial %d)", bit, got, i)
		}
	}
}

// TestKeySwitchNoiseBound checks the keyswitch error stays far inside the
// ±1/8 message space: the residual phase must stay within a few thousandths
// of the input phase.
func TestKeySwitchNoiseBound(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(6))
	outKey := NewLweKey(rnd, p)
	inKey := newBinaryKey(rnd, p.N)
	ks := p.MakeKeySwitchKey(rnd, outKey, inKey)

	var maxErr float64
	for i := 0; i < 20; i++ {
		mu := int32(rnd.Uint64()) // arbitrary phase
		in := p.encryptMu(rnd, &LweKey{Key: inKey}, mu, p.KsStdev)
		out := NewLweSample(p.LweN)
		p.KeySwitch(out, ks, in)
		errPhase := lwePhase(out, outKey) - lwePhase(in, &LweKey{Key: inKey})
		err := math.Abs(t32tod(errPhase))
		if err > maxErr {
			maxErr = err
		}
	}
	// Expected KS noise for these params: decomposition rounding
	// (~2^-17 per coefficient) plus keyswitch noise (~1e-3 total).
	// Anything under 1/64 (a quarter of the message space) proves the
	// decomposition and signs are right.
	if maxErr > 1.0/64 {
		t.Fatalf("keyswitch noise too large: %v", maxErr)
	}
	t.Logf("max keyswitch phase error: %.3e (limit 1.56e-2)", maxErr)
}

// Regression pins from the oracle review (round 1):

// phaseToT32 must use the mathematical (non-negative) remainder for ANY
// positive denominator, matching nufhe's Python semantics. Reducing
// uint32(num) first agrees only when denom divides 2^32.
func TestPhaseToT32ArbitraryDenom(t *testing.T) {
	// denom=3 does not divide 2^32: nufhe gives (-1 % 3) * (2^32//3)
	// = 2 * 1431655765 = 2863311530 -> Torus32 wrap = -1431655766.
	if got := phaseToT32(-1, 3); got != -1431655766 {
		t.Fatalf("phaseToT32(-1,3) = %d, want -1431655766 (nufhe semantics)", got)
	}
	// powers of two must be unchanged: -1/8 -> -2^29
	if got := phaseToT32(-1, 8); got != -(1 << 29) {
		t.Fatalf("phaseToT32(-1,8) = %d, want %d", got, -(1 << 29))
	}
	// negative non-coprime numerator: -2 mod 3 = 1 -> 1431655765
	if got := phaseToT32(-2, 3); got != 1431655765 {
		t.Fatalf("phaseToT32(-2,3) = %d, want 1431655765", got)
	}
}

// dtot32 must match libtfhe exactly: strip the integral part (truncation
// toward zero) BEFORE scaling by 2^32. Scaling the whole value diverges
// once d*2^32 leaves the int64 range.
func TestDtot32MatchesLibtfheDomain(t *testing.T) {
	// libtfhe: int32_t(int64_t((d - int64_t(d)) * 2^32))
	// d = 2^31 + 0.5 -> frac 0.5 -> 2^31 -> wrap = INT32_MIN
	if got := dtot32(2147483648.5); got != -2147483648 {
		t.Fatalf("dtot32(2^31+0.5) = %d, want -2147483648 (libtfhe semantics)", got)
	}
	// d = -(2^31 + 0.5): int64(d) = -2^31 (trunc toward zero), frac = -0.5
	if got := dtot32(-2147483648.5); got != -2147483648 {
		t.Fatalf("dtot32(-(2^31+0.5)) = %d, want -2147483648", got)
	}
	// in-domain behavior unchanged: 0.75 and 1.75 both map to 0.75*2^32
	if got := dtot32(0.75); got != -1073741824 {
		t.Fatalf("dtot32(0.75) = %d, want -1073741824", got)
	}
	if got := dtot32(1.75); got != -1073741824 {
		t.Fatalf("dtot32(1.75) = %d, want -1073741824", got)
	}
}

// Regression pin from the oracle review (round 2): keyswitch-key noise must
// be recentered across the whole generated keyset (libtfhe
// lweCreateKeySwitchKey "recenter the noises"; nufhe centered=True). The
// sum of body-noises over all ks entries is then bounded by int32 rounding
// only; uncentered sampling would leave a random walk ~sqrt(sizeks)*sigma.
func TestKeySwitchNoiseCentered(t *testing.T) {
	p := TestParams()
	rnd := rand.New(rand.NewSource(77))
	outKey := NewLweKey(rnd, p)
	inKey := newBinaryKey(rnd, p.LweN) // extracted key as in-key
	ks := p.MakeKeySwitchKey(rnd, outKey, inKey)
	base := 1 << p.KsBit
	var sum int64
	var count int64
	for i := 0; i < ks.inN; i++ {
		for j := 0; j < p.KsL; j++ {
			for h := 1; h < base; h++ {
				a, b := ks.at(i, j, h)
				phase := b
				for m := range a {
					phase -= a[m] * outKey.Key[m]
				}
				msg := inKey[i] * int32(h) * int32(uint32(1)<<(32-(j+1)*p.KsBit))
				sum += int64(phase - msg) // int32 wrap = noise
				count++
			}
		}
	}
	// centered: |sum| <= count/2 (rounding). uncentered std ~ sqrt(1536)*165k
	// = 6.5M LSB; 5000 is ~400x below that and ~7x above the rounding bound.
	if abs64(sum) > 5000 {
		t.Fatalf("keyswitch noise not centered: sum=%d over %d entries", sum, count)
	}
	t.Logf("centered noise sum=%d over %d entries (bound 5000)", sum, count)
}

// abs64 returns the magnitude of a bounded noise sum for regression checks.
func abs64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}
