// Torus32 helpers implement wrapping phase arithmetic, boolean encoding, exact
// rational conversion, modulus switching, and negacyclic polynomial rotations.
// Modulus switching maps LWE coefficients to the 2N rotation indices used by
// blind rotation; the same encoding helpers define the ±1/8 gate message points.
// Arithmetic wraps modulo 2^32, and conversion semantics match the nufhe/libtfhe
// references, including arbitrary positive rational denominators.
package fhebool

import "math/rand"

// Torus32 arithmetic: int32 with wrapping add/sub is exactly the torus
// mod 2^32 (see fhe/fft). Phase comparisons use the int32 sign bit.

// modSwitchFromTorus32 maps a Torus32 phase to an integer in [0, Msize),
// rounding to the nearest multiple of 2^32/Msize. Exact port of libtfhe's
// modSwitchFromTorus32.
func modSwitchFromTorus32(phase int32, Msize int) int32 {
	interv := ((uint64(1) << 63) / uint64(Msize)) * 2
	half := interv / 2
	phase64 := (uint64(uint32(phase)) << 32) + half
	return int32(phase64 / interv)
}

// phaseToT32 converts an exact rational phase num/denom to a Torus32.
// Mirrors nufhe's phase_to_t32 exactly: (num mod denom) * (2^32/denom),
// where mod is the mathematical (always non-negative) remainder for
// positive denom. The remainder must be taken in the signed domain:
// reducing uint32(num) first only agrees when denom divides 2^32.
func phaseToT32(num, denom int32) int32 {
	if denom <= 0 {
		panic("phaseToT32: denom must be positive")
	}
	r := int64(num) % int64(denom)
	if r < 0 {
		r += int64(denom)
	}
	return int32(uint32(uint64(r) * ((uint64(1) << 32) / uint64(denom))))
}

// boolToT32 encodes a gate bit as ±1/8 (nufhe's bool_to_t32).
func boolToT32(b bool) int32 {
	mu := phaseToT32(1, 8)
	if b {
		return mu
	}
	return -mu
}

// t32ToBool decodes a phase in the ±1/8 message space (nufhe's t32_to_bool).
func t32ToBool(mu int32) bool { return mu > 0 }

// mulByXai computes result = X^a * source mod (X^N+1), 0 <= a < 2N.
// Port of libtfhe torusPolynomialMulByXai.
func mulByXai(result []int32, a int, source []int32) {
	N := len(source)
	if a < N {
		for i := 0; i < a; i++ {
			result[i] = -source[i-a+N]
		}
		for i := a; i < N; i++ {
			result[i] = source[i-a]
		}
	} else {
		aa := a - N
		for i := 0; i < aa; i++ {
			result[i] = source[i-aa+N]
		}
		for i := aa; i < N; i++ {
			result[i] = -source[i-aa]
		}
	}
}

// mulByXaiMinusOne computes result = (X^a - 1) * source mod (X^N+1),
// domain 0 <= a < 2N (libtfhe asserts this; out-of-domain a is algebraically
// handled correctly here but is not part of the tested contract). result and
// source must not alias (callers use ping-pong buffers).
func mulByXaiMinusOne(result []int32, a int, source []int32) {
	N := len(source)
	if a < N {
		for i := 0; i < a; i++ {
			result[i] = -source[i-a+N] - source[i]
		}
		for i := a; i < N; i++ {
			result[i] = source[i-a] - source[i]
		}
	} else {
		aa := a - N
		for i := 0; i < aa; i++ {
			result[i] = source[i-aa+N] - source[i]
		}
		for i := aa; i < N; i++ {
			result[i] = -source[i-aa] - source[i]
		}
	}
}

// addPolyTo: a += b (coefficient-wise, wrapping).
func addPolyTo(a, b []int32) {
	for i := range a {
		a[i] += b[i]
	}
}

// subPolyTo: a -= b.
func subPolyTo(a, b []int32) {
	for i := range a {
		a[i] -= b[i]
	}
}

// randUniformTorus32 returns n uniform Torus32 values.
func randUniformTorus32(rnd *rand.Rand, n int) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(rnd.Uint64())
	}
	return out
}

// randGaussianCenteredTorus32 returns n Gaussian samples (torus-fraction
// stdev) whose float64 draws have their common mean removed BEFORE the
// Torus32 conversion. This matches libtfhe lweCreateKeySwitchKey ("recenter
// the noises") and nufhe's centered=True, used ONLY for generated keysets
// (keyswitch keys); per-sample encryption noise stays uncentered.
func randGaussianCenteredTorus32(rnd *rand.Rand, n int, stdev float64) []int32 {
	e := make([]float64, n)
	var sum float64
	for i := range e {
		e[i] = rnd.NormFloat64() * stdev
		sum += e[i]
	}
	mean := sum / float64(n)
	out := make([]int32, n)
	for i := range e {
		out[i] = dtot32(e[i] - mean)
	}
	return out
}

// randGaussianTorus32 returns n Gaussian samples with the given torus-fraction
// stdev, converted to Torus32.
func randGaussianTorus32(rnd *rand.Rand, n int, stdev float64) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = dtot32(rnd.NormFloat64() * stdev)
	}
	return out
}

// dtot32 converts a torus value to Torus32. Exact port of libtfhe dtot32:
//
//	int32_t(int64_t((d - int64_t(d)) * 2^32))
//
// i.e. drop the integral part (truncation toward zero), scale the
// fractional part by 2^32, wrap. Scaling the whole value instead agrees
// only while d*2^32 stays inside int64 and float64 precision.
func dtot32(d float64) int32 {
	return int32(int64((d - float64(int64(d))) * float64(uint64(1)<<32)))
}

// t32tod converts Torus32 to a torus fraction in [-0.5, 0.5).
func t32tod(t int32) float64 {
	return float64(t) / float64(uint32(1)<<31) / 2
}
