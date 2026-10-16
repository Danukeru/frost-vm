// This file builds FFT-resident TGSW samples and the bootstrap key consumed by
// blind rotation. Each bootstrap-key entry encrypts one LWE secret bit under the
// TLWE key; its rows are arranged as (mask/body block, decomposition level).
// The TGSW message follows the diagonal convention: row (block, level) places
// its gadget-scaled message in that polynomial block, not always in the body.
package fhebool

import (
	"math/rand"

	"github.com/flashnext/frost-vm/fhe/fft"
)

// TgswSampleFFT is a TGSW sample in the Fourier domain (TGswSampleFFT in
// libtfhe): kpl = (k+1)*l = 2l TLWE rows, each row a pair of LagrangeHalfC
// buffers. Row index q = m*l + j selects mask polynomial m (0=mask, 1=body)
// and decomposition level j, matching tGswTorus32PolynomialDecompH's
// deca[i*l+j] layout used by tGswFFTExternMulToTLwe.
type TgswSampleFFT struct {
	n    int
	data []float64 // [kpl][2][n]
}

// newTgswSampleFFT allocates zeroed Fourier buffers for all TGSW rows.
func (p *Params) newTgswSampleFFT() *TgswSampleFFT {
	return &TgswSampleFFT{n: p.N, data: make([]float64, p.kpl()*2*p.N)}
}

// row returns the Fourier buffer for row q, polynomial which (0=mask, 1=body).
func (g *TgswSampleFFT) row(q, which int) []float64 {
	off := (q*2 + which) * g.n
	return g.data[off : off+g.n]
}

// BootstrapKey is nufhe's BootstrapKey: for each i in [0, LweN), a TGSW
// sample (in FFT form) encrypting the LWE key bit key[i] under the TLWE key.
type BootstrapKey struct {
	rows []*TgswSampleFFT
}

// MakeBootstrappingKey generates the bootstrap key (BootstrapKey.from_rng +
// tgsw_transform_samples): each LWE key bit is TGSW-encrypted under the TLWE
// key with BsStdev noise, then every TLWE polynomial is transformed into the
// LagrangeHalfC domain (tGswToFFTConvert).
func (p *Params) MakeBootstrappingKey(rnd *rand.Rand, proc *fft.Processor, lweKey *LweKey, tlweKey []int32) *BootstrapKey {
	bk := &BootstrapKey{rows: make([]*TgswSampleFFT, len(lweKey.Key))}
	// Fourier form of the TLWE key polynomial (reused for every row):
	// TLWE bodies are b = a*s mod (X^N+1) + msg + noise, a polynomial product.
	sFFT := proc.NewFourierBuffer()
	proc.PolyToFourier(sFFT, tlweKey)
	aFFT := proc.NewFourierBuffer()
	bInt := make([]int32, p.N)
	for i, bit := range lweKey.Key {
		g := p.newTgswSampleFFT()
		for q := 0; q < p.kpl(); q++ {
			m := q / p.BgL // bloc: which polynomial carries the message
			j := q % p.BgL // level
			a := randUniformTorus32(rnd, p.N)
			// b = <a, s> (negacyclic convolution) + noise
			proc.PolyToFourier(aFFT, a)
			fft.FourierMulTo(aFFT, aFFT, sFFT)
			proc.FourierToPoly(bInt, aFFT)
			for k := 0; k < p.N; k++ {
				bInt[k] += randGaussianTorus32(rnd, 1, p.BsStdev)[0]
			}
			// tGswAddMuIntH: result.bloc_sample[bloc][i].a[bloc].coefsT[0] +=
			// message*h[i] — the message goes into polynomial `bloc` of row
			// (bloc, level), NOT always the body. This diagonal placement is
			// what makes the external product multiply the accumulator's
			// PHASE (b - a*s) rather than the sum of its polynomials.
			msg := bit * p.h[j]
			if m == 0 {
				a[0] += msg
			} else {
				bInt[0] += msg
			}
			proc.PolyToFourier(g.row(q, 0), a)
			proc.PolyToFourier(g.row(q, 1), bInt)
		}
		bk.rows[i] = g
	}
	return bk
}

// tlweRowCoeffs inverse-transforms one TGSW row back to coefficient domain
// (tGswFromFFTConvert); used only by tests.
func (p *Params) tlweRowCoeffs(proc *fft.Processor, g *TgswSampleFFT, q int) (a, b []int32) {
	a = make([]int32, p.N)
	b = make([]int32, p.N)
	proc.FourierToPoly(a, g.row(q, 0))
	proc.FourierToPoly(b, g.row(q, 1))
	return a, b
}
