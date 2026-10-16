// This file connects Torus32 coefficient polynomials to the LagrangeHalfC
// evaluation domain used by FFT multiplication and the FHE external product.
// Processor owns both transform tables plus odd-root and X^a-1 lookup data;
// forward and reverse buffers have exactly N values. The inverse conversion
// applies 2/N scaling and rounds to the nearest integer before Torus32 wrap,
// preserving exact integer round-trips for supported power-of-two N sizes.
package fft

import "math"

// Processor is the FHE-facing glue (port of FFT_Processor_Spqlios from
// fft_processor_spqlios.cpp): it converts between int32/Torus32 polynomials
// mod X^N+1 and the LagrangeHalfC evaluation domain used for the external
// product and KeySwitch.
type Processor struct {
	N    int // polynomial size (N in TFHE notation; logical FFT size is 2N)
	Ns2  int
	twoN int

	fftTab  *FFTTable
	ifftTab *IFFTTable

	reva           []int32 // rev(odd indices, 2N)
	cosOmegaMinus1 []float64
	sinOmegaMinus1 []float64

	scratch []float64 // N doubles, reusable for FourierToPoly
}

// NewProcessor builds forward+reverse tables and the auxiliary tables for
// polynomials of size N (power of two, >= 16).
func NewProcessor(N int) (*Processor, error) {
	if err := checkSize(N); err != nil {
		return nil, err
	}
	p := &Processor{N: N, Ns2: N / 2, twoN: 2 * N}
	var err error
	if p.fftTab, err = NewFFTTable(N); err != nil {
		return nil, err
	}
	if p.ifftTab, err = NewIFFTTable(N); err != nil {
		return nil, err
	}
	p.reva = make([]int32, p.Ns2)
	rev1 := rev(1, p.twoN)
	rev3 := rev(3, p.twoN)
	for revi := rev1; revi < rev3; revi++ {
		p.reva[revi-rev1] = int32(rev(revi, p.twoN))
	}
	p.cosOmegaMinus1 = make([]float64, p.twoN)
	p.sinOmegaMinus1 = make([]float64, p.twoN)
	for j := 0; j < p.twoN; j++ {
		p.cosOmegaMinus1[j] = math.Cos(2*math.Pi*float64(j)/float64(p.twoN)) - 1.0
		p.sinOmegaMinus1[j] = math.Sin(2 * math.Pi * float64(j) / float64(p.twoN))
	}
	p.scratch = make([]float64, N)
	return p, nil
}

// NewFourierBuffer returns a zeroed evaluation-domain buffer of length N.
func (p *Processor) NewFourierBuffer() []float64 { return make([]float64, p.N) }

// PolyToFourier transforms an integer polynomial (length N) into the
// LagrangeHalfC evaluation domain (libtfhe's IntPolynomial_ifft /
// execute_reverse_int).
func (p *Processor) PolyToFourier(dst []float64, a []int32) {
	if len(dst) != p.N || len(a) != p.N {
		panic("fft: PolyToFourier buffer/operand size mismatch")
	}
	for i, v := range a {
		dst[i] = float64(v)
	}
	p.ifftTab.IFFT(dst)
}

// FourierToPoly transforms an evaluation-domain buffer back to a Torus32
// polynomial (libtfhe's TorusPolynomial_fft / execute_direct_torus32): scale
// by 2/N, forward transform, convert to int32 (mod 2^32).
//
// Note: libtfhe truncates toward zero when converting (Torus32(int64_t(x)));
// we round to nearest instead, which is within 1 LSB of the reference and
// makes integer round-trips exact. For FHE use (torus values with noise) the
// two are equivalent.
func (p *Processor) FourierToPoly(dst []int32, src []float64) {
	if len(dst) != p.N || len(src) != p.N {
		panic("fft: FourierToPoly buffer/operand size mismatch")
	}
	const2overN := 2.0 / float64(p.N)
	s := p.scratch
	for i, v := range src {
		s[i] = v * const2overN
	}
	p.fftTab.FFT(s)
	for i, v := range s {
		// int64->int32 wraps mod 2^32, matching Torus32(int64_t(x))
		dst[i] = int32(int64(math.Round(v)))
	}
}

// FourierToPolyFloat transforms an evaluation-domain buffer to float64
// coefficients scaled by 2/N (no int32 conversion) — used for testing against
// exact references.
func (p *Processor) FourierToPolyFloat(dst []float64, src []float64) {
	if len(dst) != p.N || len(src) != p.N {
		panic("fft: FourierToPolyFloat buffer/operand size mismatch")
	}
	const2overN := 2.0 / float64(p.N)
	for i, v := range src {
		dst[i] = v * const2overN
	}
	p.fftTab.FFT(dst)
}

// PolyMulNegacyclic computes the product mod X^N+1 of two integer polynomials
// via the Fourier domain (the external-product inner loop). Result is exact
// when coefficients stay small.
func (p *Processor) PolyMulNegacyclic(dst []int32, a, b []int32) {
	fa := p.NewFourierBuffer()
	fb := p.NewFourierBuffer()
	p.PolyToFourier(fa, a)
	p.PolyToFourier(fb, b)
	FourierMulTo(fa, fa, fb)
	p.FourierToPoly(dst, fa)
}
