// This file implements binary LWE keys and samples, encryption/decryption, free
// linear operations, and KeySwitch between the extracted ring key and LWE key.
// LWE phase is b - <a,s>; encryption places the message plus small noise there.
// Linear combinations preserve the corresponding phase combination, while
// KeySwitch uses centered signed digits and pre-encrypted gadget entries to
// return bootstrap output to the original key space.
package fhebool

import "math/rand"

// LweKey is a binary secret key of size n (LweKey in nufhe/libtfhe).
type LweKey struct {
	Key []int32 // bits in {0,1}
}

// newBinaryKey samples an n-bit secret vector with independent uniform bits.
func newBinaryKey(rnd *rand.Rand, n int) []int32 {
	k := make([]int32, n)
	for i := range k {
		k[i] = int32(rnd.Intn(2))
	}
	return k
}

// NewLweKey generates a random binary LWE secret key.
func NewLweKey(rnd *rand.Rand, p *Params) *LweKey {
	return &LweKey{Key: newBinaryKey(rnd, p.LweN)}
}

// LweSample is a discrete-Gaussian LWE sample under a key of size n.
type LweSample struct {
	A []int32 // mask
	B int32   // body
}

// NewLweSample allocates a zero sample for the given key size.
func NewLweSample(n int) *LweSample {
	return &LweSample{A: make([]int32, n)}
}

// Copy returns a deep copy.
func (x *LweSample) Copy() *LweSample {
	c := &LweSample{A: make([]int32, len(x.A)), B: x.B}
	copy(c.A, x.A)
	return c
}

// lwePhase computes b - <a, s> in Torus32 (libtfhe lwePhase).
func lwePhase(x *LweSample, key *LweKey) int32 {
	phase := x.B
	for i, a := range x.A {
		phase -= a * key.Key[i]
	}
	return phase
}

// EncryptBit encrypts a gate bit (±1/8 message space) under the secret key
// with the keyswitch stdev (nufhe Context.encrypt: noise = in_out_params.min_noise).
func (p *Params) EncryptBit(rnd *rand.Rand, key *LweKey, bit bool) *LweSample {
	return p.encryptMu(rnd, key, boolToT32(bit), p.KsStdev)
}

// encryptMu forms an LWE sample whose phase is mu plus Gaussian noise.
func (p *Params) encryptMu(rnd *rand.Rand, key *LweKey, mu int32, stdev float64) *LweSample {
	n := len(key.Key)
	a := randUniformTorus32(rnd, n)
	b := mu
	for i := 0; i < n; i++ {
		b += a[i] * key.Key[i]
	}
	b += randGaussianTorus32(rnd, 1, stdev)[0]
	return &LweSample{A: a, B: b}
}

// DecryptBit decrypts a sample in the ±1/8 message space (nufhe decrypt:
// phase -> phase > 0).
func (p *Params) DecryptBit(key *LweKey, x *LweSample) bool {
	return t32ToBool(lwePhase(x, key))
}

// The lwe*To helpers apply signed, wrapping linear arithmetic to mask/body
// pairs, so their ciphertext phases undergo the same addition or subtraction.
// Gate preparation uses these free combinations; KeySwitch accumulates samples
// by subtraction.
func lweAddTo(result, y *LweSample) {
	for i := range result.A {
		result.A[i] += y.A[i]
	}
	result.B += y.B
}

func lweSubTo(result, y *LweSample) {
	for i := range result.A {
		result.A[i] -= y.A[i]
	}
	result.B -= y.B
}

func lweAddMulTo(result *LweSample, m int32, y *LweSample) {
	for i := range result.A {
		result.A[i] += m * y.A[i]
	}
	result.B += m * y.B
}

func lweSubMulTo(result *LweSample, m int32, y *LweSample) {
	for i := range result.A {
		result.A[i] -= m * y.A[i]
	}
	result.B -= m * y.B
}

// lweNegate: result = -y (nufhe gate_not).
func lweNegate(result, y *LweSample) {
	for i := range result.A {
		result.A[i] = -y.A[i]
	}
	result.B = -y.B
}

// noiselessTrivialConstant: result = (0, mu).
func noiselessTrivialConstant(result *LweSample, mu int32) {
	for i := range result.A {
		result.A[i] = 0
	}
	result.B = mu
}

// ---------------------------------------------------------------------------
// KeySwitch
// ---------------------------------------------------------------------------

// KeySwitchKey translates messages from an input key of size inN to an output
// key of size p.LweN. ks[i][j][k] encrypts k*inKey[i]/Bg'^(j+1) under the
// output key. Layout: data[((i*KsL + j)*base + k)*(inN... no: outN+1 slots)].
type KeySwitchKey struct {
	inN, outN int
	base, ksL int
	data      []int32 // [inN][KsL][base] * (outN+1): A(outN) then B
}

// at returns the aliased mask slice and body for gadget entry (i,j,k).
func (ks *KeySwitchKey) at(i, j, k int) ([]int32, int32) {
	idx := ks.index(i, j, k)
	return ks.data[idx : idx+ks.outN], ks.data[idx+ks.outN]
}

// MakeKeySwitchKey fills the keyswitch key: ks[i][j][k] = LWE_enc(outKey,
// k*inKey[i]*(2^(32-(j+1)*basebit)), ksStdev) (libtfhe
// lweCreateKeySwitchKey_fromArray; ks[i][j][0] is the zero sample).
func (p *Params) MakeKeySwitchKey(rnd *rand.Rand, outKey *LweKey, inKey []int32) *KeySwitchKey {
	base := 1 << p.KsBit
	ks := &KeySwitchKey{
		inN:  len(inKey),
		outN: p.LweN,
		base: base,
		ksL:  p.KsL,
		data: make([]int32, len(inKey)*p.KsL*base*(p.LweN+1)),
	}
	// libtfhe lweCreateKeySwitchKey: one noise vector of sizeks = n*t*(base-1)
	// draws, recentered (mean removed) across the whole keyset, consumed in
	// (i, j, h=1..base-1) order. k=0 entries are trivial zero encryptions
	// ("it will not be used in the KeySwitching").
	noises := randGaussianCenteredTorus32(rnd, len(inKey)*p.KsL*(base-1), p.KsStdev)
	index := 0
	for i := 0; i < len(inKey); i++ {
		for j := 0; j < p.KsL; j++ {
			for k := 0; k < base; k++ {
				idx := ks.index(i, j, k)
				if k == 0 {
					continue // zero sample
				}
				msg := inKey[i] * int32(k) * (int32(uint32(1) << (32 - (j+1)*p.KsBit)))
				a := randUniformTorus32(rnd, p.LweN)
				b := msg
				for m := 0; m < p.LweN; m++ {
					b += a[m] * outKey.Key[m]
				}
				b += noises[index]
				index++
				copy(ks.data[idx:idx+p.LweN], a)
				ks.data[idx+p.LweN] = b
			}
		}
	}
	return ks
}

// index computes the flat offset of gadget entry (i,j,k) in key data.
func (ks *KeySwitchKey) index(i, j, k int) int {
	return ((i*ks.ksL+j)*ks.base + k) * (ks.outN + 1)
}

// KeySwitch translates a sample under the input key (size ks.inN) to one
// under the output key. Exact port of nufhe's LweKeyswitch kernel:
// result = (0, b) - sum_{i,j} ks[i][j][digit_j(a_i + ksOffset)].
func (p *Params) KeySwitch(result *LweSample, ks *KeySwitchKey, x *LweSample) {
	base := int32(1) << p.KsBit
	mask := uint32(base - 1)
	for i := range result.A {
		result.A[i] = 0
	}
	result.B = x.B
	for i := 0; i < ks.inN; i++ {
		tmp := uint32(x.A[i]) + p.ksOffset
		for j := 0; j < p.KsL; j++ {
			val := (tmp >> (32 - (j+1)*p.KsBit)) & mask
			if val == 0 {
				continue
			}
			ka, kb := ks.at(i, j, int(val))
			for m := range result.A {
				result.A[m] -= ka[m]
			}
			result.B -= kb
		}
	}
}
