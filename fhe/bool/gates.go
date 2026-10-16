// Context owns the LWE/TLWE keys and cloud keys used by the boolean gate API.
// Binary gates form a free LWE linear combination, then bootstrap its phase;
// NOT and CONSTANT remain free, while MUX uses two bootstraps without keyswitch
// followed by one addition and a final keyswitch. Gate inputs must encode ±1/8
// with noise below 1/16 of the torus, as produced by EncryptBit or a gate output.
package fhebool

import (
	"fmt"
	"math/rand"

	"github.com/flashnext/frost-vm/fhe/fft"
)

// Context mirrors nufhe's api_high_level.Context: a secret key, the derived
// cloud keys (keyswitch + bootstrap), and the boolean gate API.
//
// Gate contract (as in nufhe/libtfhe): every gate input must be a
// ciphertext produced by this same Context (or its gate outputs) encoding a
// bit as ±1/8 with noise below 1/16 of the torus — i.e. an EncryptBit
// output or another gate's output. Inputs outside that message space and
// noise bound may decrypt to the wrong bit.
type Context struct {
	Params  *Params
	proc    *fft.Processor
	key     *LweKey
	tlweKey []int32
	ks      *KeySwitchKey
	bk      *BootstrapKey
	rnd     *rand.Rand
}

// NewContext generates a fresh key pair. seed makes the whole context
// (keys, noise) deterministic; pass 0 and a fresh time-based seed via
// NewContextRandom for production use.
func NewContext(seed int64) (*Context, error) {
	return NewContextWithParams(DefaultParams(), seed)
}

// NewContextWithParams builds a context for an explicit parameter set.
func NewContextWithParams(p *Params, seed int64) (*Context, error) {
	proc, err := fft.NewProcessor(p.N)
	if err != nil {
		return nil, fmt.Errorf("frost-vm: fft processor: %w", err)
	}
	rnd := rand.New(rand.NewSource(seed))
	c := &Context{Params: p, proc: proc, rnd: rnd}
	c.key = NewLweKey(rnd, p)
	c.tlweKey = newBinaryKey(rnd, p.N)
	// keyswitch: extracted key (TLWE key viewed as LWE key of size N) -> LWE key
	c.ks = p.MakeKeySwitchKey(rnd, c.key, c.tlweKey)
	c.bk = p.MakeBootstrappingKey(rnd, proc, c.key, c.tlweKey)
	return c, nil
}

// NewCiphertext allocates an empty ciphertext in the input key space.
func (c *Context) NewCiphertext() *LweSample { return NewLweSample(c.Params.LweN) }

// newExtractedCiphertext allocates an empty sample in the extracted key
// space (size N), used by bootstraps without keyswitch (MUX path).
func (c *Context) newExtractedCiphertext() *LweSample {
	return NewLweSample(c.Params.N)
}

// EncryptBit encrypts a bit under the secret key (nufhe Context.encrypt).
func (c *Context) EncryptBit(bit bool) *LweSample {
	return c.Params.EncryptBit(c.rnd, c.key, bit)
}

// DecryptBit decrypts a ciphertext (nufhe Context.decrypt).
func (c *Context) DecryptBit(x *LweSample) bool {
	return c.Params.DecryptBit(c.key, x)
}

// trivialConstant: result = (0, mu) (lwe_noiseless_trivial_constant).
func trivialConstant(result *LweSample, mu int32) {
	noiselessTrivialConstant(result, mu)
}

// ---------------------------------------------------------------------------
// Gates — exact ports of nufhe gates.py (constants via phase_to_t32).
// ---------------------------------------------------------------------------

const (
	mu1_8 = int32(1 << 29) // phase_to_t32(1, 8)
	mu1_4 = int32(1 << 30) // phase_to_t32(1, 4)
)

// bootstrapInto runs the full bootstrap (with keyswitch) into a sample in
// the input key space.
func (c *Context) bootstrapInto(result *LweSample, mu int32, x *LweSample) {
	sc := c.Params.newScratch()
	c.Params.blindRotateAndExtract(result, mu, x, c.bk, c.ks, c.proc, sc, false)
}

// bootstrapNoKS writes the extracted N-dimensional sample without keyswitch.
func (c *Context) bootstrapNoKS(result *LweSample, mu int32, x *LweSample) {
	sc := c.Params.newScratch()
	c.Params.blindRotateAndExtract(result, mu, x, c.bk, c.ks, c.proc, sc, true)
}

// GateNand: compute (0,1/8) - a - b, bootstrap with MU=1/8.
func (c *Context) GateNand(result, a, b *LweSample) {
	temp := c.NewCiphertext()
	trivialConstant(temp, mu1_8)
	lweSubTo(temp, a)
	lweSubTo(temp, b)
	c.bootstrapInto(result, mu1_8, temp)
}

// GateAnd: (0,-1/8) + a + b, bootstrap MU=1/8.
func (c *Context) GateAnd(result, a, b *LweSample) {
	temp := c.NewCiphertext()
	trivialConstant(temp, -mu1_8)
	lweAddTo(temp, a)
	lweAddTo(temp, b)
	c.bootstrapInto(result, mu1_8, temp)
}

// GateOr: (0,1/8) + a + b, bootstrap MU=1/8.
func (c *Context) GateOr(result, a, b *LweSample) {
	temp := c.NewCiphertext()
	trivialConstant(temp, mu1_8)
	lweAddTo(temp, a)
	lweAddTo(temp, b)
	c.bootstrapInto(result, mu1_8, temp)
}

// GateNor: (0,-1/8) - a - b, bootstrap MU=1/8.
func (c *Context) GateNor(result, a, b *LweSample) {
	temp := c.NewCiphertext()
	trivialConstant(temp, -mu1_8)
	lweSubTo(temp, a)
	lweSubTo(temp, b)
	c.bootstrapInto(result, mu1_8, temp)
}

// GateXor: (0,1/4) + 2a + 2b, bootstrap MU=1/8.
func (c *Context) GateXor(result, a, b *LweSample) {
	temp := c.NewCiphertext()
	trivialConstant(temp, mu1_4)
	lweAddMulTo(temp, 2, a)
	lweAddMulTo(temp, 2, b)
	c.bootstrapInto(result, mu1_8, temp)
}

// GateXnor: (0,-1/4) - 2a - 2b, bootstrap MU=1/8.
func (c *Context) GateXnor(result, a, b *LweSample) {
	temp := c.NewCiphertext()
	trivialConstant(temp, -mu1_4)
	lweSubMulTo(temp, 2, a)
	lweSubMulTo(temp, 2, b)
	c.bootstrapInto(result, mu1_8, temp)
}

// GateAndNY: (0,-1/8) - a + b ("(not a) and b"), bootstrap MU=1/8.
func (c *Context) GateAndNY(result, a, b *LweSample) {
	temp := c.NewCiphertext()
	trivialConstant(temp, -mu1_8)
	lweSubTo(temp, a)
	lweAddTo(temp, b)
	c.bootstrapInto(result, mu1_8, temp)
}

// GateAndYN: (0,-1/8) + a - b ("a and (not b)"), bootstrap MU=1/8.
func (c *Context) GateAndYN(result, a, b *LweSample) {
	temp := c.NewCiphertext()
	trivialConstant(temp, -mu1_8)
	lweAddTo(temp, a)
	lweSubTo(temp, b)
	c.bootstrapInto(result, mu1_8, temp)
}

// GateOrNY: (0,1/8) - a + b ("(not a) or b"), bootstrap MU=1/8.
func (c *Context) GateOrNY(result, a, b *LweSample) {
	temp := c.NewCiphertext()
	trivialConstant(temp, mu1_8)
	lweSubTo(temp, a)
	lweAddTo(temp, b)
	c.bootstrapInto(result, mu1_8, temp)
}

// GateOrYN: (0,1/8) + a - b ("a or (not b)"), bootstrap MU=1/8.
func (c *Context) GateOrYN(result, a, b *LweSample) {
	temp := c.NewCiphertext()
	trivialConstant(temp, mu1_8)
	lweAddTo(temp, a)
	lweSubTo(temp, b)
	c.bootstrapInto(result, mu1_8, temp)
}

// GateNot: result = -a (not bootstrapped).
func (c *Context) GateNot(result, a *LweSample) {
	lweNegate(result, a)
}

// GateConstant: trivial encryption of a plaintext bit (gate_constant).
func (c *Context) GateConstant(result *LweSample, bit bool) {
	noiselessTrivialConstant(result, boolToT32(bit))
}

// GateMux computes result = b if a else c (nufhe gate_mux: the SELECTOR is
// the first ciphertext argument a, matching nufhe's gate_mux(a, b, c)).
// Two bootstraps without keyswitch, sum with (0,1/8), then a single
// keyswitch.
func (c *Context) GateMux(result, a, b, cval *LweSample) {
	mu := mu1_8
	u1 := c.newExtractedCiphertext()
	u2 := c.newExtractedCiphertext()
	temp := c.NewCiphertext()

	// "AND(a,b)": (0,-1/8) + a + b
	trivialConstant(temp, -mu)
	lweAddTo(temp, a)
	lweAddTo(temp, b)
	c.bootstrapNoKS(u1, mu, temp)

	// "AND(not(a),c)": (0,-1/8) - a + c
	trivialConstant(temp, -mu)
	lweSubTo(temp, a)
	lweAddTo(temp, cval)
	c.bootstrapNoKS(u2, mu, temp)

	// result = (0,1/8) + u1 + u2, keyswitched
	temp1 := c.newExtractedCiphertext()
	trivialConstant(temp1, mu)
	lweAddTo(temp1, u1)
	lweAddTo(temp1, u2)
	c.Params.KeySwitch(result, c.ks, temp1)
}
