// TLWE samples are the ring-side ciphertexts used as the bootstrap accumulator.
// This file supplies their trivial initialization, arithmetic, phase helper, and
// coefficient-zero extraction back to an LWE sample. The ring is Z[X]/(X^N+1),
// with one mask polynomial and one body polynomial (k=1); extraction negates
// reversed mask coefficients so the resulting phase matches the ring phase.
package fhebool

// TLWE samples over Z_N[X]/(X^N+1) with mask_size k=1 (nufhe default):
// one mask polynomial A[0] and one body polynomial B.

// TlweSample is a TLWE sample (k=1).
type TlweSample struct {
	A [][]int32 // A[0]: mask polynomial
	B []int32   // body polynomial
}

// newTlweSample allocates a zero k=1 TLWE sample with degree-N polynomials.
func (p *Params) newTlweSample() *TlweSample {
	return &TlweSample{
		A: [][]int32{make([]int32, p.N)},
		B: make([]int32, p.N),
	}
}

// poly returns the q-th polynomial of the sample: q=0 is the mask, q=1 the
// body (matches libtfhe's x->a[i] indexing where a[k] is the body).
func (s *TlweSample) poly(q int) []int32 {
	if q == 0 {
		return s.A[0]
	}
	return s.B
}

// noiselessTrivial: result = (0, mu) (tLweNoiselessTrivial).
func (p *Params) noiselessTrivial(result *TlweSample, mu []int32) {
	for i := range result.A[0] {
		result.A[0][i] = 0
	}
	copy(result.B, mu)
}

// tlweAddTo: result += y (tLweAddTo).
func (p *Params) tlweAddTo(result, y *TlweSample) {
	addPolyTo(result.A[0], y.A[0])
	addPolyTo(result.B, y.B)
}

// tlwePhasePoly computes the phase polynomial b - a*s mod (X^N+1) of a TLWE
// sample under the binary TLWE key, in Torus32 coefficients. Used only by
// tests.
func (p *Params) tlwePhasePoly(x *TlweSample, key []int32) []int32 {
	out := make([]int32, p.N)
	copy(out, x.B)
	// out -= A[0] * s  (negacyclic convolution)
	s := key
	a := x.A[0]
	for i, av := range a {
		if av == 0 {
			continue
		}
		for j, sj := range s {
			if sj == 0 {
				continue
			}
			k := i + j
			if k >= p.N {
				k -= p.N
				out[k] += av // -(-av) : negacyclic double flip
				continue
			}
			out[k] -= av
		}
	}
	return out
}

// extractLwe copies the accumulator into an LWE sample of size N
// (tLweExtractLweSampleIndex with index=0, k=1):
//
//	result.a[j] =  x.a[0][0-j]        for j = 0
//	result.a[j] = -x.a[0][N-j]        for j = 1..N-1
//	result.b   =  x.b[0]
func (p *Params) extractLwe(result *LweSample, x *TlweSample) {
	result.A[0] = x.A[0][0]
	for j := 1; j < p.N; j++ {
		result.A[j] = -x.A[0][p.N-j]
	}
	result.B = x.B[0]
}
