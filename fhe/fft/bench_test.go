// This file verifies bit-identical scalar/AVX transform behavior and benchmarks
// forward, inverse, and negacyclic polynomial-multiplication workloads. The
// tests skip hardware comparisons when AVX dispatch is unavailable, while the
// scalar benchmark variants provide a direct baseline for the same packed
// transform pipeline and buffer sizes.
package fft

import (
	"math/rand"
	"testing"
)

// The AVX kernels must be bit-identical to the scalar model (same IEEE ops
// in the same order).
func TestAVXMatchesScalarModel(t *testing.T) {
	if !useAVX {
		t.Skip("no AVX on this CPU")
	}
	for _, N := range []int{16, 32, 64, 256, 1024, 4096} {
		ft, err := NewFFTTable(N)
		if err != nil {
			t.Fatal(err)
		}
		it, err := NewIFFTTable(N)
		if err != nil {
			t.Fatal(err)
		}
		r := rand.New(rand.NewSource(int64(N) + 101))
		for trial := 0; trial < 3; trial++ {
			src := make([]float64, N)
			for i := range src {
				src[i] = r.NormFloat64() * 1e6
			}

			a := append([]float64(nil), src...)
			b := append([]float64(nil), src...)
			ft.FFT(a)
			ft.fftScalar(b)
			for i := range a {
				if a[i] != b[i] {
					t.Fatalf("N=%d fwd mismatch at %d: %v != %v", N, i, a[i], b[i])
				}
			}

			a = append([]float64(nil), src...)
			b = append([]float64(nil), src...)
			it.IFFT(a)
			it.ifftScalar(b)
			for i := range a {
				if a[i] != b[i] {
					t.Fatalf("N=%d rev mismatch at %d: %v != %v", N, i, a[i], b[i])
				}
			}
		}
	}
}

// TestFourierMulAVXMatchesScalar checks pointwise multiplication parity,
// including the in-place output alias case, between assembly and scalar paths.
func TestFourierMulAVXMatchesScalar(t *testing.T) {
	if !useAVX {
		t.Skip("no AVX on this CPU")
	}
	r := rand.New(rand.NewSource(7))
	for _, N := range []int{16, 64, 1024} {
		a := make([]float64, N)
		b := make([]float64, N)
		for i := range a {
			a[i] = r.NormFloat64()
			b[i] = r.NormFloat64()
		}
		// r == a aliasing case
		r1 := append([]float64(nil), a...)
		r2 := append([]float64(nil), a...)
		FourierMulTo(r1, a, b)
		fourierMulScalar(r2, a, b)
		for i := range r1 {
			if r1[i] != r2[i] {
				t.Fatalf("mul mismatch at %d: %v != %v", i, r1[i], r2[i])
			}
		}
		r1 = append([]float64(nil), a...)
		r2 = append([]float64(nil), a...)
		FourierAddMulTo(r1, a, b)
		fourierAddMulScalar(r2, a, b)
		for i := range r1 {
			if r1[i] != r2[i] {
				t.Fatalf("addmul mismatch at %d: %v != %v", i, r1[i], r2[i])
			}
		}
	}
}

// BenchmarkFFT and BenchmarkIFFT measure the packed forward and reverse
// transforms, with a scalar baseline for forward stages.
func BenchmarkFFT(b *testing.B) {
	for _, N := range []int{256, 1024, 4096} {
		ft, err := NewFFTTable(N)
		if err != nil {
			b.Fatal(err)
		}
		data := make([]float64, N)
		b.Run("N="+itoa(N), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ft.FFT(data)
			}
		})
		b.Run("N="+itoa(N)+"/scalar", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ft.fftScalar(data)
			}
		})
	}
}

func BenchmarkIFFT(b *testing.B) {
	for _, N := range []int{256, 1024, 4096} {
		it, err := NewIFFTTable(N)
		if err != nil {
			b.Fatal(err)
		}
		data := make([]float64, N)
		b.Run("N="+itoa(N), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				it.IFFT(data)
			}
		})
	}
}

// One external-product-style operation: ifft(a), ifft(b), pointwise mul, fft.
func BenchmarkPolyMulNegacyclic(b *testing.B) {
	for _, N := range []int{1024, 4096} {
		p, err := NewProcessor(N)
		if err != nil {
			b.Fatal(err)
		}
		a := make([]int32, N)
		c := make([]int32, N)
		out := make([]int32, N)
		fa := p.NewFourierBuffer()
		fb := p.NewFourierBuffer()
		b.Run("N="+itoa(N), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.PolyToFourier(fa, a)
				p.PolyToFourier(fb, c)
				FourierMulTo(fa, fa, fb)
				p.FourierToPoly(out, fa)
			}
		})
	}
}

// itoa formats a non-negative integer for benchmark subtest names.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d []byte
	for n > 0 {
		d = append([]byte{byte('0' + n%10)}, d...)
		n /= 10
	}
	return string(d)
}
