// This file implements pointwise operations on the LagrangeHalfC half-spectrum,
// where each buffer stores real values followed by imaginary values. Complex
// multiplication here is negacyclic polynomial multiplication and feeds the
// external-product pipeline; constant and X^a-1 helpers construct matching
// spectra directly. Public multiply operations dispatch to AVX when available,
// while scalar fallbacks preserve the same lane order and alias-safe products.
package fft

// LagrangeHalfC pointwise operations on the evaluation-domain representation
// (port of lagrangehalfc_impl.cpp / lagrangehalfc_impl_avx.s).
//
// A buffer of length N holds N/2 complex numbers: re = data[:N/2],
// im = data[N/2:].

// FourierClear sets the representation to zero.
func FourierClear(data []float64) {
	for i := range data {
		data[i] = 0
	}
}

// FourierAddTo accumulates a into accum (accum += a).
func FourierAddTo(accum, a []float64) {
	for i := range accum {
		accum[i] += a[i]
	}
}

// FourierMulTo computes r = a * b pointwise (complex multiply).
func FourierMulTo(r, a, b []float64) {
	if useAVX {
		fourierMulAVX(&r[0], &a[0], &b[0], len(r))
		return
	}
	fourierMulScalar(r, a, b)
}

// fourierMulScalar and fourierAddMulScalar are the scalar fallback pair for
// the AVX-dispatched pointwise operations; both preserve SoA order, and the
// multiply path computes all products before storing to allow r to alias a.
func fourierMulScalar(r, a, b []float64) {
	ns2 := len(r) / 2
	rr, ri := r[:ns2], r[ns2:]
	ar, ai := a[:ns2], a[ns2:]
	br, bi := b[:ns2], b[ns2:]
	for i := 0; i < ns2; i++ {
		// all products before any store: r may alias a
		t0 := ar[i] * br[i]
		t1 := ai[i] * bi[i]
		t2 := ar[i] * bi[i]
		t3 := ai[i] * br[i]
		rr[i] = t0 - t1
		ri[i] = t2 + t3
	}
}

// FourierAddMulTo computes r += a * b pointwise.
func FourierAddMulTo(r, a, b []float64) {
	if useAVX {
		fourierAddMulAVX(&r[0], &a[0], &b[0], len(r))
		return
	}
	fourierAddMulScalar(r, a, b)
}

func fourierAddMulScalar(r, a, b []float64) {
	ns2 := len(r) / 2
	rr, ri := r[:ns2], r[ns2:]
	ar, ai := a[:ns2], a[ns2:]
	br, bi := b[:ns2], b[ns2:]
	for i := 0; i < ns2; i++ {
		t0 := ar[i] * br[i]
		t1 := ai[i] * bi[i]
		t2 := ar[i] * bi[i]
		t3 := ai[i] * br[i]
		rr[i] += t0 - t1
		ri[i] += t2 + t3
	}
}

// FourierSubMulTo computes r -= a * b pointwise.
func FourierSubMulTo(r, a, b []float64) {
	ns2 := len(r) / 2
	rr, ri := r[:ns2], r[ns2:]
	ar, ai := a[:ns2], a[ns2:]
	br, bi := b[:ns2], b[ns2:]
	for i := 0; i < ns2; i++ {
		t0 := ar[i] * br[i]
		t1 := ai[i] * bi[i]
		t2 := ar[i] * bi[i]
		t3 := ai[i] * br[i]
		rr[i] -= t0 - t1
		ri[i] -= t2 + t3
	}
}

// FourierSetTorusConstant sets the representation of the constant polynomial
// mu (re = mu, im = 0).
func FourierSetTorusConstant(data []float64, mu float64) {
	ns2 := len(data) / 2
	for i := 0; i < ns2; i++ {
		data[i] = mu
	}
	for i := ns2; i < len(data); i++ {
		data[i] = 0
	}
}

// FourierAddTorusConstant adds the constant polynomial mu (re += mu).
func FourierAddTorusConstant(data []float64, mu float64) {
	ns2 := len(data) / 2
	for i := 0; i < ns2; i++ {
		data[i] += mu
	}
}

// FourierSetXaiMinusOne sets the representation to (omega^x - 1) evaluated at
// the odd roots, i.e. X^ai - 1 in the LagrangeHalfC domain (port of
// LagrangeHalfCPolynomialSetXaiMinusOne).
func (p *Processor) FourierSetXaiMinusOne(data []float64, ai int) {
	ns2 := p.Ns2
	twoNm1 := p.twoN - 1
	for i := 0; i < ns2; i++ {
		ii := (int(p.reva[i]) * ai) & twoNm1
		data[i] = p.cosOmegaMinus1[ii]
		data[ns2+i] = p.sinOmegaMinus1[ii]
	}
}
