//go:build amd64

#include "textflag.h"

// This file implements the amd64 AVX assembly kernels for the packed FFT and
// LagrangeHalfC pointwise operations. Forward and inverse butterflies consume
// four-lane SoA blocks and interleaved cosine/sine tables; multiply kernels
// operate on the real and imaginary halves of evaluation-domain buffers.
// Runtime dispatch is guarded by CPUID/XGETBV, and arithmetic order matches
// the scalar model to preserve bit-identical results.

// func hasAVX() bool
TEXT ·hasAVX(SB), NOSPLIT, $0-1
	MOVL	$1, AX
	CPUID
	// ECX bit 27 = OSXSAVE, bit 28 = AVX
	ANDL	$0x18000000, CX
	CMPL	CX, $0x18000000
	JNE	nov
	MOVL	$0, CX
	XGETBV
	// XMM and YMM state must be OS-enabled (bits 1,2 of EAX)
	ANDL	$6, AX
	CMPL	AX, $6
	JNE	nov
	MOVB	$1, ret+0(FP)
	RET
nov:
	MOVB	$0, ret+0(FP)
	RET

// func fftStageAVX(data *float64, trig *float64, ns4, halfnn, ttOff int)
//
// R8=pre base, DI=pim base, R9=trig+ttOff*8, R15=ns4*8 (byte bound),
// R14=halfnn*8 (byte half-step), SI=2*halfnn*8 (byte step),
// AX=block byte offset, DX=off byte offset.
TEXT ·fftStageAVX(SB), NOSPLIT, $0-40
	MOVQ	data+0(FP), R8
	MOVQ	trig+8(FP), R9
	MOVQ	ns4+16(FP), R15
	MOVQ	halfnn+24(FP), R14
	MOVQ	ttOff+32(FP), CX
	SHLQ	$3, CX
	ADDQ	CX, R9
	MOVQ	R15, DI
	SHLQ	$3, DI
	ADDQ	R8, DI
	SHLQ	$3, R15
	MOVQ	R14, SI
	SHLQ	$4, SI
	SHLQ	$3, R14
	XORQ	AX, AX
fouter:
	XORQ	DX, DX
finner:
	MOVQ	R8, R10
	ADDQ	AX, R10
	ADDQ	DX, R10		// re0
	MOVQ	DI, R11
	ADDQ	AX, R11
	ADDQ	DX, R11		// im0
	LEAQ	(R10)(R14*1), R12	// re1
	LEAQ	(R11)(R14*1), R13	// im1
	LEAQ	(R9)(DX*2), CX		// tcs

	VMOVDQU	(R12), Y0		// re1
	VMOVDQU	(R13), Y1		// im1
	VMOVDQU	(CX), Y2		// tcs
	VMOVDQU	32(CX), Y3		// tsn
	VMULPD	Y2, Y0, Y4		// re*cos
	VMULPD	Y3, Y0, Y5		// re*sin
	VMULPD	Y2, Y1, Y6		// im*cos
	VMULPD	Y3, Y1, Y7		// im*sin
	VSUBPD	Y7, Y4, Y4		// re2
	VADDPD	Y6, Y5, Y5		// im2
	VMOVDQU	(R10), Y8		// re0
	VMOVDQU	(R11), Y9		// im0
	VADDPD	Y4, Y8, Y6		// re0+re2
	VADDPD	Y5, Y9, Y7		// im0+im2
	VSUBPD	Y4, Y8, Y4		// re0-re2
	VSUBPD	Y5, Y9, Y5		// im0-im2
	VMOVDQU	Y6, (R10)
	VMOVDQU	Y7, (R11)
	VMOVDQU	Y4, (R12)
	VMOVDQU	Y5, (R13)

	ADDQ	$32, DX
	CMPQ	DX, R14
	JB	finner
	ADDQ	SI, AX
	CMPQ	AX, R15
	JB	fouter
	VZEROUPPER
	RET

// func twiddleMulAVX(data *float64, trig *float64, ns4, ttOff int)
// shared by the forward final stage and the reverse first stage.
TEXT ·twiddleMulAVX(SB), NOSPLIT, $0-32
	MOVQ	data+0(FP), R8
	MOVQ	trig+8(FP), R9
	MOVQ	ns4+16(FP), R15
	MOVQ	ttOff+24(FP), CX
	SHLQ	$3, CX
	ADDQ	CX, R9
	MOVQ	R15, DI
	SHLQ	$3, DI
	ADDQ	R8, DI
	SHLQ	$3, R15
	XORQ	AX, AX
tloop:
	MOVQ	R8, R10
	ADDQ	AX, R10		// d0
	MOVQ	DI, R11
	ADDQ	AX, R11		// d1
	LEAQ	(R9)(AX*2), CX	// r0

	VMOVDQU	(R10), Y0		// d0
	VMOVDQU	(R11), Y1		// d1
	VMOVDQU	(CX), Y2		// r0
	VMOVDQU	32(CX), Y3		// r1
	VMULPD	Y2, Y0, Y4		// re*cos
	VMULPD	Y2, Y1, Y5		// im*cos
	VMULPD	Y3, Y0, Y6		// re*sin
	VMULPD	Y3, Y1, Y7		// im*sin
	VSUBPD	Y7, Y4, Y4		// d0'
	VADDPD	Y6, Y5, Y5		// d1'
	VMOVDQU	Y4, (R10)
	VMOVDQU	Y5, (R11)

	ADDQ	$32, AX
	CMPQ	AX, R15
	JB	tloop
	VZEROUPPER
	RET

// func ifftStageAVX(data *float64, trig *float64, ns4, halfnn, ttOff int)
TEXT ·ifftStageAVX(SB), NOSPLIT, $0-40
	MOVQ	data+0(FP), R8
	MOVQ	trig+8(FP), R9
	MOVQ	ns4+16(FP), R15
	MOVQ	halfnn+24(FP), R14
	MOVQ	ttOff+32(FP), CX
	SHLQ	$3, CX
	ADDQ	CX, R9
	MOVQ	R15, DI
	SHLQ	$3, DI
	ADDQ	R8, DI
	SHLQ	$3, R15
	MOVQ	R14, SI
	SHLQ	$4, SI
	SHLQ	$3, R14
	XORQ	AX, AX
rioter:
	XORQ	DX, DX
rinne:
	MOVQ	R8, R10
	ADDQ	AX, R10
	ADDQ	DX, R10		// d00
	MOVQ	DI, R11
	ADDQ	AX, R11
	ADDQ	DX, R11		// d01
	LEAQ	(R10)(R14*1), R12	// d10
	LEAQ	(R11)(R14*1), R13	// d11
	LEAQ	(R9)(DX*2), CX		// r0

	VMOVDQU	(R10), Y0		// d00
	VMOVDQU	(R11), Y1		// d01
	VMOVDQU	(R12), Y2		// d10
	VMOVDQU	(R13), Y3		// d11
	VADDPD	Y2, Y0, Y6		// re+re
	VADDPD	Y3, Y1, Y7		// im+im
	VSUBPD	Y2, Y0, Y8		// re-re
	VSUBPD	Y3, Y1, Y9		// im-im
	VMOVDQU	Y6, (R10)
	VMOVDQU	Y7, (R11)
	VMOVDQU	(CX), Y4		// r0
	VMOVDQU	32(CX), Y5		// r1
	VMULPD	Y4, Y8, Y10		// (re-re)*cos
	VMULPD	Y5, Y9, Y11		// (im-im)*sin
	VSUBPD	Y11, Y10, Y2		// d10'
	VMULPD	Y5, Y8, Y10		// (re-re)*sin
	VMULPD	Y4, Y9, Y11		// (im-im)*cos
	VADDPD	Y11, Y10, Y3		// d11'
	VMOVDQU	Y2, (R12)
	VMOVDQU	Y3, (R13)

	ADDQ	$32, DX
	CMPQ	DX, R14
	JB	rinne
	ADDQ	SI, AX
	CMPQ	AX, R15
	JB	rioter
	VZEROUPPER
	RET

// func fourierMulAVX(r *float64, a *float64, b *float64, n int)
// r = a*b pointwise (complex, SoA). Handles r aliasing a or b: all loads
// before any store.
TEXT ·fourierMulAVX(SB), NOSPLIT, $0-32
	MOVQ	r+0(FP), R8
	MOVQ	a+8(FP), R9
	MOVQ	b+16(FP), R10
	MOVQ	n+24(FP), R15
	SHLQ	$2, R15			// (n/2)*8 byte bound
	MOVQ	R9, R11
	ADDQ	R15, R11		// ai
	MOVQ	R10, R13
	ADDQ	R15, R13		// bi
	MOVQ	R8, R14
	ADDQ	R15, R14		// ri
	XORQ	AX, AX
mloop:
	VMOVDQU	(R9)(AX*1), Y0	// ar
	VMOVDQU	(R11)(AX*1), Y1	// ai
	VMOVDQU	(R10)(AX*1), Y2	// br
	VMOVDQU	(R13)(AX*1), Y3	// bi
	VMULPD	Y2, Y0, Y4		// t0
	VMULPD	Y3, Y1, Y5		// t1
	VMULPD	Y3, Y0, Y6		// t2
	VMULPD	Y2, Y1, Y7		// t3
	VSUBPD	Y5, Y4, Y4		// rr
	VADDPD	Y6, Y7, Y7		// ri
	VMOVDQU	Y4, (R8)(AX*1)
	VMOVDQU	Y7, (R14)(AX*1)
	ADDQ	$32, AX
	CMPQ	AX, R15
	JB	mloop
	VZEROUPPER
	RET

// func fourierAddMulAVX(r *float64, a *float64, b *float64, n int)
// r += a*b pointwise.
TEXT ·fourierAddMulAVX(SB), NOSPLIT, $0-32
	MOVQ	r+0(FP), R8
	MOVQ	a+8(FP), R9
	MOVQ	b+16(FP), R10
	MOVQ	n+24(FP), R15
	SHLQ	$2, R15
	MOVQ	R9, R11
	ADDQ	R15, R11
	MOVQ	R10, R13
	ADDQ	R15, R13
	MOVQ	R8, R14
	ADDQ	R15, R14
	XORQ	AX, AX
amloop:
	VMOVDQU	(R9)(AX*1), Y0
	VMOVDQU	(R11)(AX*1), Y1
	VMOVDQU	(R10)(AX*1), Y2
	VMOVDQU	(R13)(AX*1), Y3
	VMULPD	Y2, Y0, Y4		// t0
	VMULPD	Y3, Y1, Y5		// t1
	VMULPD	Y3, Y0, Y6		// t2
	VMULPD	Y2, Y1, Y7		// t3
	VMOVDQU	(R8)(AX*1), Y8	// rr
	VMOVDQU	(R14)(AX*1), Y9	// ri
	VSUBPD	Y5, Y4, Y4		// t0-t1
	VADDPD	Y6, Y7, Y7		// t2+t3
	VADDPD	Y4, Y8, Y4		// rr +=
	VADDPD	Y7, Y9, Y7		// ri +=
	VMOVDQU	Y4, (R8)(AX*1)
	VMOVDQU	Y7, (R14)(AX*1)
	ADDQ	$32, AX
	CMPQ	AX, R15
	JB	amloop
	VZEROUPPER
	RET
