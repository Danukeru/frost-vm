// Package fft implements the LagrangeHalfC Fourier machinery used by TFHE-style
// FHE (the algorithm family implemented by nufhe / libtfhe), ported from the
// spqlios backend of libtfhe (https://github.com/tfhe/tfhe,
// src/libtfhe/fft_processors/spqlios/).
//
// A real polynomial P mod X^N+1 is represented as the N/2 complex numbers
// P(w), P(w^3), ..., P(w^(N-1)) with w = exp(i*pi/N) — evaluations at the odd
// 2N-th roots of unity, stored in bit-reversed order as two SoA halves
// (real parts, then imaginary parts). Pointwise complex multiplication in this
// domain is negacyclic polynomial multiplication mod X^N+1, which is the core
// operation of the TFHE external product (blind rotation) and KeySwitch.
//
// The transform itself is the spqlios real-input FFT of size 2N packed into
// N/2 complex values. The scalar model ported here (spqlios.go) is the exact
// specification of the hand-written AVX kernels (spqlios-fft-avx.s etc.); the
// 4-wide SoA block structure maps 1:1 onto ymm (4x double) SIMD kernels.
// This file builds the forward and inverse trigonometric tables consumed by
// the packed FFT stages in spqlios.go and the amd64 assembly kernels. Entries
// are grouped as four cosines followed by four sines for each four-lane step,
// matching the kernels' SoA loads without gathers. Range-reduced trig helpers
// preserve the reference table values, and accepted polynomial sizes are
// powers of two of at least 16.
package fft

import (
	"errors"
	"math"
)

// accurateCos returns cos(2*pi*i/n) computed with the same range reduction as
// libtfhe's accurate_cos, so table values are bit-identical to the reference.
func accurateCos(i, n int) float64 {
	i = ((i % n) + n) % n
	switch {
	case i >= 3*n/4:
		return math.Cos(2.0 * math.Pi * float64(n-i) / float64(n))
	case i >= 2*n/4:
		return -math.Cos(2.0 * math.Pi * float64(i-n/2) / float64(n))
	case i >= 1*n/4:
		return -math.Cos(2.0 * math.Pi * float64(n/2-i) / float64(n))
	default:
		return math.Cos(2.0 * math.Pi * float64(i) / float64(n))
	}
}

// accurateSin returns sin(2*pi*i/n) (same range reduction as accurate_sin).
func accurateSin(i, n int) float64 {
	i = ((i % n) + n) % n
	switch {
	case i >= 3*n/4:
		return -math.Sin(2.0 * math.Pi * float64(n-i) / float64(n))
	case i >= 2*n/4:
		return -math.Sin(2.0 * math.Pi * float64(i-n/2) / float64(n))
	case i >= 1*n/4:
		return math.Sin(2.0 * math.Pi * float64(n/2-i) / float64(n))
	default:
		return math.Sin(2.0 * math.Pi * float64(i) / float64(n))
	}
}

// rev reverses the low log2(m) bits of x (libtfhe's rev()).
func rev(x, m int) int {
	reps := 0
	for j := m; j > 1; j /= 2 {
		reps = 2*reps + (x % 2)
		x /= 2
	}
	return reps
}

// checkSize validates the polynomial size accepted by the Fourier tables.
func checkSize(nn int) error {
	if nn < 16 {
		return errors.New("fft: polynomial size N must be >= 16")
	}
	if nn&(nn-1) != 0 {
		return errors.New("fft: polynomial size N must be a power of 2")
	}
	return nil
}

// FFTTable holds the precomputed trig tables for the forward transform
// (coefficient/half-spectrum domain -> evaluation domain), i.e. libtfhe's
// new_fft_table. The logical FFT length is n = 2*nn; data buffers passed to
// FFT have length nn (= n/2 packed doubles).
type FFTTable struct {
	n    int // logical real-FFT size (2*nn)
	trig []float64
}

// NewFFTTable builds the forward tables for polynomials of size nn (N in TFHE
// notation). nn must be a power of two >= 16.
func NewFFTTable(nn int) (*FFTTable, error) {
	if err := checkSize(nn); err != nil {
		return nil, err
	}
	n := 2 * nn
	ns4 := n / 4
	t := &FFTTable{n: n, trig: make([]float64, n)}
	ptr := 0
	// subsequent iterations
	for halfnn := 4; halfnn < ns4; halfnn *= 2 {
		nnl := 2 * halfnn
		j := n / nnl
		for i := 0; i < halfnn; i += 4 {
			for k := 0; k < 4; k++ {
				t.trig[ptr] = accurateCos(-j*(i+k), n)
				ptr++
			}
			for k := 0; k < 4; k++ {
				t.trig[ptr] = accurateSin(-j*(i+k), n)
				ptr++
			}
		}
	}
	// last iteration
	for i := 0; i < ns4; i += 4 {
		for k := 0; k < 4; k++ {
			t.trig[ptr] = accurateCos(-(i + k), n)
			ptr++
		}
		for k := 0; k < 4; k++ {
			t.trig[ptr] = accurateSin(-(i + k), n)
			ptr++
		}
	}
	return t, nil
}

// IFFTTable holds the precomputed trig tables for the reverse transform
// (evaluation domain -> coefficient domain), i.e. libtfhe's new_ifft_table.
type IFFTTable struct {
	n    int
	trig []float64
}

// NewIFFTTable builds the reverse tables for polynomials of size nn.
func NewIFFTTable(nn int) (*IFFTTable, error) {
	if err := checkSize(nn); err != nil {
		return nil, err
	}
	n := 2 * nn
	ns4 := n / 4
	t := &IFFTTable{n: n, trig: make([]float64, n)}
	ptr := 0
	// first iteration
	for j := 0; j < ns4; j += 4 {
		for k := 0; k < 4; k++ {
			t.trig[ptr] = accurateCos(j+k, n)
			ptr++
		}
		for k := 0; k < 4; k++ {
			t.trig[ptr] = accurateSin(j+k, n)
			ptr++
		}
	}
	// subsequent iterations
	for nnl := ns4; nnl >= 8; nnl /= 2 {
		halfnn := nnl / 2
		j := n / nnl
		for i := 0; i < halfnn; i += 4 {
			for k := 0; k < 4; k++ {
				t.trig[ptr] = accurateCos(j*(i+k), n)
				ptr++
			}
			for k := 0; k < 4; k++ {
				t.trig[ptr] = accurateSin(j*(i+k), n)
				ptr++
			}
		}
	}
	return t, nil
}
