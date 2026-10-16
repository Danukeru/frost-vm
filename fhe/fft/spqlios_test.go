// This file checks the Fourier core's representation and arithmetic contracts:
// exact Torus32 round-trips, direct evaluation at odd roots, and negacyclic
// products against independent references. It also pins constant, shift, and
// trig-table semantics that the Processor and FFT stages rely on. Tests use
// fixed random seeds so failures are reproducible across scalar and AVX paths.
package fft

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// ---- helpers -------------------------------------------------------------

// randPoly generates a deterministic test polynomial from the supplied RNG.
func randPoly(r *rand.Rand, n int, small bool) []int32 {
	a := make([]int32, n)
	for i := range a {
		if small {
			a[i] = int32(r.Intn(1000) - 500)
		} else {
			a[i] = int32(r.Uint32())
		}
	}
	return a
}

// naive negacyclic product mod X^N+1 in exact int64 arithmetic
func negacyclicRef(a, b []int32) []int64 {
	n := len(a)
	c := make([]int64, n)
	for i := 0; i < n; i++ {
		if a[i] == 0 {
			continue
		}
		for j := 0; j < n; j++ {
			k := i + j
			if k >= n {
				c[k-n] -= int64(a[i]) * int64(b[j])
			} else {
				c[k] += int64(a[i]) * int64(b[j])
			}
		}
	}
	return c
}

// ---- tests ---------------------------------------------------------------

// Mirrors libtfhe's LagrangeHalfcTest.fftIsBijective: random full-range
// Torus32 polynomials must survive PolyToFourier -> FourierToPoly exactly.
func TestFFTIsBijective(t *testing.T) {
	for _, N := range []int{16, 32, 64, 256, 1024, 4096} {
		p, err := NewProcessor(N)
		if err != nil {
			t.Fatal(err)
		}
		r := rand.New(rand.NewSource(int64(N)))
		for trial := 0; trial < 3; trial++ {
			a := randPoly(r, N, false)
			f := p.NewFourierBuffer()
			p.PolyToFourier(f, a)
			b := make([]int32, N)
			p.FourierToPoly(b, f)
			for i := range a {
				if a[i] != b[i] {
					t.Fatalf("N=%d trial=%d: round-trip mismatch at %d: %d != %d", N, trial, i, a[i], b[i])
				}
			}
		}
	}
}

// Independent check of the documented LagrangeHalfC semantics: the buffer
// holds P(w^r) for odd r = reva[i], w = exp(i*pi/N).
func TestFourierIsEvaluationAtOddRoots(t *testing.T) {
	for _, N := range []int{16, 64, 1024} {
		p, err := NewProcessor(N)
		if err != nil {
			t.Fatal(err)
		}
		r := rand.New(rand.NewSource(int64(N) + 7))
		a := randPoly(r, N, true)
		f := p.NewFourierBuffer()
		p.PolyToFourier(f, a)
		w := cmplx.Exp(complex(0, math.Pi/float64(N))) // exp(i*pi/N)
		for i := 0; i < p.Ns2; i++ {
			rIdx := int(p.reva[i])
			// direct evaluation P(w^rIdx)
			acc := complex(0, 0)
			wp := cmplx.Pow(w, complex(float64(rIdx), 0))
			step := complex(1, 0)
			for k := 0; k < N; k++ {
				acc += complex(float64(a[k]), 0) * step
				step *= wp
			}
			got := complex(f[i], f[p.Ns2+i])
			scale := float64(N) * 500 // bound on |P|
			if diff := cmplx.Abs(got - acc); diff > 1e-6*scale {
				t.Fatalf("N=%d slot=%d (root w^%d): got %v want %v (diff %g)", N, i, rIdx, got, acc, diff)
			}
		}
	}
}

// The Fourier-domain product must equal the exact negacyclic product.
func TestNegacyclicMulMatchesReference(t *testing.T) {
	for _, N := range []int{16, 64, 256, 1024} {
		p, err := NewProcessor(N)
		if err != nil {
			t.Fatal(err)
		}
		r := rand.New(rand.NewSource(int64(N) + 11))
		for trial := 0; trial < 3; trial++ {
			a := randPoly(r, N, true)
			b := randPoly(r, N, true)
			want := negacyclicRef(a, b)
			got := make([]int32, N)
			p.PolyMulNegacyclic(got, a, b)
			for i := range want {
				if int64(got[i]) != want[i] {
					t.Fatalf("N=%d trial=%d: product mismatch at %d: got %d want %d", N, trial, i, got[i], want[i])
				}
			}
		}
	}
}

// Mirrors LagrangeHalfCPolynomialClear: zero Fourier -> zero poly.
func TestFourierClear(t *testing.T) {
	p, err := NewProcessor(1024)
	if err != nil {
		t.Fatal(err)
	}
	f := p.NewFourierBuffer()
	FourierClear(f)
	out := make([]int32, p.N)
	p.FourierToPoly(out, f)
	for i, v := range out {
		if v != 0 {
			t.Fatalf("clear: nonzero at %d: %d", i, v)
		}
	}
}

// Mirrors LagrangeHalfCPolynomialSetTorusConstant.
func TestFourierSetTorusConstant(t *testing.T) {
	p, err := NewProcessor(1024)
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewSource(42))
	for trial := 0; trial < 3; trial++ {
		mu := int32(r.Uint32())
		f := p.NewFourierBuffer()
		FourierSetTorusConstant(f, float64(mu))
		out := make([]int32, p.N)
		p.FourierToPoly(out, f)
		if out[0] != mu {
			t.Fatalf("const: out[0]=%d want %d", out[0], mu)
		}
		for i := 1; i < p.N; i++ {
			if out[i] != 0 {
				t.Fatalf("const: nonzero at %d: %d", i, out[i])
			}
		}
	}
}

// Mirrors LagrangeHalfCPolynomialAddTorusConstant: adds mu to coefficient 0.
func TestFourierAddTorusConstant(t *testing.T) {
	p, err := NewProcessor(1024)
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewSource(43))
	for trial := 0; trial < 3; trial++ {
		mu := int32(r.Uint32())
		a := randPoly(r, p.N, false)
		f := p.NewFourierBuffer()
		p.PolyToFourier(f, a)
		FourierAddTorusConstant(f, float64(mu))
		out := make([]int32, p.N)
		p.FourierToPoly(out, f)
		want0 := a[0] + mu // torus arithmetic = int32 wraparound
		if out[0] != want0 {
			t.Fatalf("addconst: out[0]=%d want %d", out[0], want0)
		}
		for i := 1; i < p.N; i++ {
			if out[i] != a[i] {
				t.Fatalf("addconst: mismatch at %d: %d != %d", i, out[i], a[i])
			}
		}
	}
}

// Mirrors LagrangeHalfCPolynomialAddTo.
func TestFourierAddTo(t *testing.T) {
	p, err := NewProcessor(1024)
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewSource(44))
	for trial := 0; trial < 3; trial++ {
		a := randPoly(r, p.N, false)
		b := randPoly(r, p.N, false)
		fa := p.NewFourierBuffer()
		fb := p.NewFourierBuffer()
		p.PolyToFourier(fa, a)
		p.PolyToFourier(fb, b)
		FourierAddTo(fa, fb)
		out := make([]int32, p.N)
		p.FourierToPoly(out, fa)
		for i := range out {
			want := a[i] + b[i] // int32 wraparound
			if out[i] != want {
				t.Fatalf("addto: mismatch at %d: %d != %d", i, out[i], want)
			}
		}
	}
}

// X^ai - 1 in the Fourier domain, times a polynomial, must equal the
// negacyclic product (X^ai - 1) * b.
func TestSetXaiMinusOne(t *testing.T) {
	p, err := NewProcessor(1024)
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewSource(45))
	for trial := 0; trial < 3; trial++ {
		ai := r.Intn(p.N)
		b := randPoly(r, p.N, true)
		fx := p.NewFourierBuffer()
		p.FourierSetXaiMinusOne(fx, ai)
		fb := p.NewFourierBuffer()
		p.PolyToFourier(fb, b)
		FourierMulTo(fx, fx, fb)
		got := make([]int32, p.N)
		p.FourierToPoly(got, fx)
		// reference: (X^ai * b - b) mod X^N+1
		want := make([]int64, p.N)
		for j := 0; j < p.N; j++ {
			k := j + ai
			if k >= p.N {
				want[k-p.N] -= int64(b[j])
			} else {
				want[k] += int64(b[j])
			}
			want[j] -= int64(b[j])
		}
		for i := range want {
			if int64(got[i]) != want[i] {
				t.Fatalf("xai-1: ai=%d mismatch at %d: got %d want %d", ai, i, got[i], want[i])
			}
		}
	}
}

// Trig tables must match direct high-precision evaluation.
func TestTablesMatchDirectTrig(t *testing.T) {
	for _, N := range []int{16, 1024} {
		n := 2 * N
		ft, err := NewFFTTable(N)
		if err != nil {
			t.Fatal(err)
		}
		it, err := NewIFFTTable(N)
		if err != nil {
			t.Fatal(err)
		}
		// spot-check the last forward stage (interleaved 4+4 blocks):
		// cos(-j), sin(-j) at offset 2*(ns4-4) + (j/4)*8 + (j%4)
		ns4 := n / 4
		base := 2 * (ns4 - 4)
		for j := 0; j < ns4; j++ {
			idx := base + (j/4)*8 + (j % 4)
			wantC := math.Cos(2 * math.Pi * float64(j) / float64(n)) // cos even
			wantS := -math.Sin(2 * math.Pi * float64(j) / float64(n))
			if got := ft.trig[idx]; math.Abs(got-wantC) > 1e-15 {
				t.Fatalf("N=%d fwd trig[%d] (j=%d): %v vs %v", N, idx, j, got, wantC)
			}
			if got := ft.trig[idx+4]; math.Abs(got-wantS) > 1e-15 {
				t.Fatalf("N=%d fwd trig[%d] (j=%d sin): %v vs %v", N, idx+4, j, got, wantS)
			}
		}
		// first reverse stage: cos(j), sin(j), interleaved 4+4 blocks
		for j := 0; j < ns4; j++ {
			idx := (j/4)*8 + (j % 4)
			wantC := math.Cos(2 * math.Pi * float64(j) / float64(n))
			wantS := math.Sin(2 * math.Pi * float64(j) / float64(n))
			if got := it.trig[idx]; math.Abs(got-wantC) > 1e-15 {
				t.Fatalf("N=%d rev trig[%d] (j=%d): %v vs %v", N, idx, j, got, wantC)
			}
			if got := it.trig[idx+4]; math.Abs(got-wantS) > 1e-15 {
				t.Fatalf("N=%d rev trig[%d] (j=%d sin): %v vs %v", N, idx+4, j, got, wantS)
			}
		}
	}
}

// reva must enumerate the odd indices (the roots of X^N+1).
func TestRevaIsOddRoots(t *testing.T) {
	p, err := NewProcessor(1024)
	if err != nil {
		t.Fatal(err)
	}
	seen := make([]bool, p.twoN)
	for _, r := range p.reva {
		if r%2 == 0 || r >= int32(p.twoN) || seen[r] {
			t.Fatalf("reva: bad root %d", r)
		}
		seen[r] = true
	}
	if count := len(p.reva); count != p.Ns2 {
		t.Fatalf("reva length %d", count)
	}
}
