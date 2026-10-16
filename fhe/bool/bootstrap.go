// This file implements signed gadget decomposition, the FFT external product,
// and blind-rotate/extract with optional keyswitch. Modulus-switched LWE masks
// drive rotations of the initialized test polynomial; each TGSW product selects
// the encrypted key bit, leaving the test-vector value at the encrypted phase.
// The TGSW diagonal message convention and negacyclic X^(2N) rotation domain
// are invariants; no-key-switch output remains in the extracted N-dimensional key.
package fhebool

import (
	"math/rand"

	"github.com/flashnext/frost-vm/fhe/fft"
)

// scratch holds the reusable work buffers for one bootstrap (libtfhe's
// per-call new/delete in tGswFFTExternMulToTLwe and
// tfhe_blindRotateAndExtract_FFT, hoisted out of the loop).
// Not safe for concurrent use; one scratch per bootstrap call.
type scratch struct {
	deca    [][]int32   // [kpl] integer digit polynomials
	decaFFT [][]float64 // [kpl] their Fourier forms
	tmpa    [][]float64 // [k+1=2] Fourier accumulator
	tmp     *TlweSample // ping-pong partner of the accumulator
	polyBuf []int32     // test-vector temporaries
}

// newScratch allocates the reusable decomposition, Fourier, and ping-pong buffers
// required by one bootstrap; callers must not share it across concurrent calls.
func (p *Params) newScratch() *scratch {
	sc := &scratch{}
	sc.deca = make([][]int32, p.kpl())
	sc.decaFFT = make([][]float64, p.kpl())
	for q := range sc.deca {
		sc.deca[q] = make([]int32, p.N)
		sc.decaFFT[q] = make([]float64, p.N)
	}
	sc.tmpa = [][]float64{make([]float64, p.N), make([]float64, p.N)}
	sc.tmp = p.newTlweSample()
	sc.polyBuf = make([]int32, p.N)
	return sc
}

// decompH fills result with the centered base-Bg digit decomposition of the
// Torus32 polynomial sample at level j (tGswTorus32PolynomialDecompH):
//
//	buf = uint32(sample[i]) + offset
//	digits[i] = ((buf >> (32-(j+1)*BgBit)) & (Bg-1)) - Bg/2
func (p *Params) decompH(result []int32, sample []int32, j int) {
	shift := uint(32 - (j+1)*p.BgBit)
	for i, v := range sample {
		buf := uint32(v) + p.offset
		result[i] = int32((buf>>shift)&p.mask) - p.halfBg
	}
}

// externMulTo replaces accum with the external product
// accum (x) gsw (tGswFFTExternMulToTLwe): decompose every accumulator
// polynomial into TGSW rows, multiply pointwise in the Fourier domain,
// sum, and transform back.
func (p *Params) externMulTo(accum *TlweSample, gsw *TgswSampleFFT, proc *fft.Processor, sc *scratch) {
	// deca + ifft (PolyToFourier in our naming) per mask/body polynomial
	for q := 0; q < 2; q++ {
		src := accum.poly(q)
		for j := 0; j < p.BgL; j++ {
			d := sc.deca[q*p.BgL+j]
			p.decompH(d, src, j)
			proc.PolyToFourier(sc.decaFFT[q*p.BgL+j], d)
		}
	}
	// tmpa = sum_q decaFFT[q] * gsw.row(q)
	for m := 0; m < 2; m++ {
		fft.FourierClear(sc.tmpa[m])
	}
	for q := 0; q < p.kpl(); q++ {
		for m := 0; m < 2; m++ {
			fft.FourierAddMulTo(sc.tmpa[m], sc.decaFFT[q], gsw.row(q, m))
		}
	}
	// accum = fft(tmpa)
	for m := 0; m < 2; m++ {
		proc.FourierToPoly(accum.poly(m), sc.tmpa[m])
	}
}

// blindRotateAndExtract is the core of bootstrapping
// (tfhe_blindRotateAndExtract_FFT + the optional keyswitch):
// result = LWE(v_p) where p = barb - sum(bara_i.s_i) mod 2N.
//
// With noKeySwitch the result sample must be sized N (the extracted key
// space); otherwise it is sized LweN and keyswitched (tfhe_bootstrap_FFT).
func (p *Params) blindRotateAndExtract(result *LweSample, mu int32, x *LweSample, bk *BootstrapKey, ks *KeySwitchKey, proc *fft.Processor, sc *scratch, noKeySwitch bool) {
	twoN := 2 * p.N

	// Modulus switching (modSwitchFromTorus32)
	barb := int(modSwitchFromTorus32(x.B, twoN))
	bara := make([]int32, len(x.A))
	for i := range bara {
		bara[i] = modSwitchFromTorus32(x.A[i], twoN)
	}

	// testvect = [mu]*N ; testvectbis = X^{2N-barb} * testvect
	tv := sc.polyBuf
	for i := range tv {
		tv[i] = mu
	}
	acc := p.newTlweSample()
	a := (twoN - barb) % twoN
	if a == 0 {
		p.noiselessTrivial(acc, tv)
	} else {
		shifted := make([]int32, p.N)
		mulByXai(shifted, a, tv)
		p.noiselessTrivial(acc, shifted)
	}

	// Blind rotation (tfhe_blindRotate_FFT, ping-pong, skip barai==0)
	cur, other := acc, sc.tmp
	for i := range bara {
		if bara[i] == 0 {
			continue
		}
		for m := 0; m < 2; m++ {
			mulByXaiMinusOne(other.poly(m), int(bara[i]), cur.poly(m))
		}
		p.externMulTo(other, bk.rows[i], proc, sc)
		p.tlweAddTo(other, cur)
		cur, other = other, cur
	}

	// Extraction (tLweExtractLweSample)
	extracted := NewLweSample(p.N)
	p.extractLwe(extracted, cur)

	if noKeySwitch {
		copy(result.A, extracted.A)
		result.B = extracted.B
		return
	}
	p.KeySwitch(result, ks, extracted)
}

// bootstrap is tfhe_bootstrap_FFT: blind-rotate-and-extract followed by
// keyswitch back into the input key space.
func (p *Params) bootstrap(result *LweSample, mu int32, x *LweSample, bk *BootstrapKey, ks *KeySwitchKey, proc *fft.Processor, rnd *rand.Rand) {
	_ = rnd // noise comes from the keys, not fresh sampling (deterministic given keys)
	sc := p.newScratch()
	p.blindRotateAndExtract(result, mu, x, bk, ks, proc, sc, false)
}
