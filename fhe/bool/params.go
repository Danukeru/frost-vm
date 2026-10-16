// Package fhebool implements the TFHE boolean bootstrapping pipeline:
// LWE/TLWE/TGSW samples, KeySwitch, blind rotation via FFT-domain external
// products (using the fhe/fft LagrangeHalfC core), and the boolean gate set
// (NAND/AND/OR/NOR/XOR/XNOR/ANDNY/ANDYN/ORN/ORYN/MUX, plus NOT/CONSTANT)
// exactly as nufhe's gates.py defines them (nufhe has no NANDY).
//
// Parameter defaults match nufhe's NuFHEParameters (~128-bit security):
// N=1024, n=500, bootstrap l=2/Bg=2^10, keyswitch l'=8/Bg'=2^2.
//
// The algorithmic reference is libtfhe's FFT path:
//
//	tfhe_bootstrap_FFT / tfhe_blindRotateAndExtract_FFT /
//	tGswFFTExternMulToTLwe / tGswTorus32PolynomialDecompH /
//	tLweExtractLweSampleIndex / lweKeySwitch.
//
// Parameter sets define the LWE and ring dimensions, decomposition bases, and noise
// scales consumed by encryption, keyswitch, and bootstrapping. init derives the
// gadget weights and rounding constants shared by those pipeline stages. The
// production ring degree is a power of two (N >= 16), and decomposition settings
// must fit within 32 Torus32 bits; TestParams keeps N=1024 and reduces only LweN.
package fhebool

import "math"

// Params holds the scheme parameters and their derived constants.
type Params struct {
	N    int // TLWE polynomial degree (power of two, >= 16)
	LweN int // LWE secret key size

	// Bootstrap (TGSW) decomposition.
	BgL   int // decomposition levels
	BgBit int // log2 of decomposition base
	KsL   int // keyswitch decomposition levels
	KsBit int // log2 of the keyswitch gadget base (Bg' = 2^KsBit)

	// Noise standard deviations (Torus32 units are implicit: stdev is a
	// fraction of the torus, converted when sampling).
	KsStdev float64
	BsStdev float64

	// Derived (TGswParams in libtfhe).
	h      []int32 // h[j] = 2^(32-(j+1)*BgBit): 1/Bg^(j+1) as Torus32
	offset uint32  // rounding offset for the decomposition
	mask   uint32  // Bg-1
	halfBg int32   // Bg/2

	// Derived (keyswitch).
	ksOffset uint32 // 2^(31 - KsL*KsBit): rounding offset for KS decomposition
}

// DefaultParams returns the nufhe default parameter set.
func DefaultParams() *Params {
	p := &Params{
		N:    1024,
		LweN: 500,
		BgL:  2, BgBit: 10,
		KsL: 8, KsBit: 2,
	}
	coeff := math.Sqrt(2 / math.Pi)
	p.KsStdev = 1.0 / (1 << 15) * coeff
	p.BsStdev = 9e-9 * coeff
	p.init()
	return p
}

// TestParams returns a valid parameter set for unit tests: the production
// polynomial degree N=1024 is kept (real-degree coverage; noise budgets
// depend on N), only the LWE key size is reduced (LweN=64 vs 500) so a
// full gate-truth-table suite runs in ~1s. (N must stay a power of two
// >= 16; the decomposition still needs (BgL+1)*BgBit < 32.)
func TestParams() *Params {
	p := &Params{
		N:    1024,
		LweN: 64,
		BgL:  2, BgBit: 10,
		KsL: 8, KsBit: 2,
	}
	coeff := math.Sqrt(2 / math.Pi)
	p.KsStdev = 1.0 / (1 << 15) * coeff
	p.BsStdev = 9e-9 * coeff
	p.init()
	return p
}

// init derives bootstrap gadget weights and rounding constants for the
// configured decomposition levels, bases, and keyswitch precision.
func (p *Params) init() {
	bg := int32(1) << p.BgBit
	p.h = make([]int32, p.BgL)
	var sum uint32
	for j := 0; j < p.BgL; j++ {
		p.h[j] = int32(uint32(1) << (32 - (j+1)*p.BgBit))
		sum += uint32(1) << (32 - (j+1)*p.BgBit)
	}
	p.offset = sum * uint32(bg/2)
	p.mask = uint32(bg - 1)
	p.halfBg = bg / 2
	p.ksOffset = uint32(1) << (31 - p.KsL*p.KsBit)
}

// kpl is the number of TLWE rows in a TGSW sample: (k+1)*l with k=1.
func (p *Params) kpl() int { return 2 * p.BgL }
