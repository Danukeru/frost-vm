//go:build amd64

// This file declares the amd64 assembly kernels that accelerate the packed
// FFT stages and LagrangeHalfC pointwise products. Runtime AVX availability
// gates all dispatch through useAVX, leaving the scalar model as fallback on
// unsupported processors. Kernel pointers address the same real-then-imaginary
// SoA buffers and four-lane trig blocks used by the scalar implementation.
package fft

// hasAVX reports whether the operating system and processor support AVX state.
//
//go:noescape
func hasAVX() bool

// fftStageAVX applies one forward general butterfly stage (halfnn >= 4):
// for block in [0,ns4) step 2*halfnn, for off in [0,halfnn) step 4.
// ttOff is the element offset into trig for this stage's tables.
//
//go:noescape
func fftStageAVX(data *float64, trig *float64, ns4, halfnn, ttOff int)

// twiddleMulAVX multiplies the packed half-spectrum by the twiddle table at
// element offset ttOff (final forward stage / first reverse stage).
//
//go:noescape
func twiddleMulAVX(data *float64, trig *float64, ns4, ttOff int)

// ifftStageAVX applies one reverse general butterfly stage (halfnn = nn/2).
//
//go:noescape
func ifftStageAVX(data *float64, trig *float64, ns4, halfnn, ttOff int)

// fourierMulAVX: r = a*b; fourierAddMulAVX: r += a*b (complex pointwise,
// SoA layout, n = full buffer length).
//
//go:noescape
func fourierMulAVX(r *float64, a *float64, b *float64, n int)

//go:noescape
func fourierAddMulAVX(r *float64, a *float64, b *float64, n int)

// useAVX selects assembly kernels only when runtime AVX state is available.
var useAVX = hasAVX()
