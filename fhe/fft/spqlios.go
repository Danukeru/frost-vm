// This file implements the packed real-input forward and inverse FFT stages
// used to move between polynomial coefficients and the half-spectrum. It is
// the scalar reference path and mirrors the four-lane operation order of the
// amd64 AVX kernels. Transform buffers are in-place, have length n/2, and use
// real and imaginary SoA halves; table offsets and stage widths must remain
// aligned with the matching trig layout and assembly kernels.
package fft

// Direct port of fft_model / ifft_model from libtfhe's
// spqlios-fft-impl.cpp. Operation order is preserved exactly in both the
// scalar stages and the AVX kernels (kernel_amd64.s), so the two paths are
// bit-identical — asserted by TestAVXMatchesScalarModel.
//
// Data layout: a single buffer of length nn = n/2 doubles, interpreted as two
// SoA halves: re = data[:n/4], im = data[n/4:n/2].

// FFT applies the forward transform (evaluation-domain -> coefficient-side),
// as in libtfhe. data must have length t.n/2. Operates in place.
func (t *FFTTable) FFT(data []float64) {
	ns4 := t.n / 4
	if len(data) != 2*ns4 {
		panic("fft: FFT data buffer must have length n/2")
	}
	t.fftStage2(data)
	t.fftStage4(data)
	curTT := 0
	for halfnn := 4; halfnn < ns4; halfnn *= 2 {
		if useAVX {
			fftStageAVX(&data[0], &t.trig[0], ns4, halfnn, curTT)
		} else {
			t.fftGeneralStageScalar(data, halfnn, curTT)
		}
		curTT += 2 * halfnn
	}
	if useAVX {
		twiddleMulAVX(&data[0], &t.trig[0], ns4, curTT)
	} else {
		t.fftTwiddleScalar(data, curTT)
	}
}

// IFFT applies the reverse transform. data must have length t.n/2. In place.
func (t *IFFTTable) IFFT(data []float64) {
	ns4 := t.n / 4
	if len(data) != 2*ns4 {
		panic("fft: IFFT data buffer must have length n/2")
	}
	if useAVX {
		twiddleMulAVX(&data[0], &t.trig[0], ns4, 0)
	} else {
		t.ifftTwiddleScalar(data)
	}
	curTT := 0
	for nn := ns4; nn >= 8; nn /= 2 {
		curTT += 2 * nn
		if useAVX {
			ifftStageAVX(&data[0], &t.trig[0], ns4, nn/2, curTT)
		} else {
			t.ifftGeneralStageScalar(data, nn, curTT)
		}
	}
	t.ifftStage4(data)
	t.ifftStage2(data)
}

// ---- forward stages ------------------------------------------------------

// fftScalar runs the entire forward transform through the scalar model
// (reference path; used by tests to assert the AVX kernels are bit-exact).
func (t *FFTTable) fftScalar(data []float64) {
	ns4 := t.n / 4
	t.fftStage2(data)
	t.fftStage4(data)
	curTT := 0
	for halfnn := 4; halfnn < ns4; halfnn *= 2 {
		t.fftGeneralStageScalar(data, halfnn, curTT)
		curTT += 2 * halfnn
	}
	t.fftTwiddleScalar(data, curTT)
}

// ifftScalar runs the entire reverse transform through the scalar model.
func (t *IFFTTable) ifftScalar(data []float64) {
	ns4 := t.n / 4
	t.ifftTwiddleScalar(data)
	curTT := 0
	for nn := ns4; nn >= 8; nn /= 2 {
		curTT += 2 * nn
		t.ifftGeneralStageScalar(data, nn, curTT)
	}
	t.ifftStage4(data)
	t.ifftStage2(data)
}

// size 2
// [1  1]
// [1 -1]
//
//	[1  1]
//	[1 -1]
func (t *FFTTable) fftStage2(data []float64) {
	ns4 := t.n / 4
	pre := data[:ns4]
	pim := data[ns4:]
	for block := 0; block < ns4; block += 4 {
		d0 := pre[block : block+4]
		d1 := pim[block : block+4]
		var tmp0, tmp1 [4]float64
		tmp0[0] = d0[0]
		tmp0[1] = d0[0]
		tmp0[2] = d0[2]
		tmp0[3] = d0[2]
		tmp1[0] = d0[1]
		tmp1[1] = -d0[1]
		tmp1[2] = d0[3]
		tmp1[3] = -d0[3]
		for i := 0; i < 4; i++ {
			d0[i] = tmp0[i] + tmp1[i]
		}
		tmp0[0] = d1[0]
		tmp0[1] = d1[0]
		tmp0[2] = d1[2]
		tmp0[3] = d1[2]
		tmp1[0] = d1[1]
		tmp1[1] = -d1[1]
		tmp1[2] = d1[3]
		tmp1[3] = -d1[3]
		for i := 0; i < 4; i++ {
			d1[i] = tmp0[i] + tmp1[i]
		}
	}
}

// size 4
// [1  0  1  0]
// [0  1  0 -i]
// [1  0 -1  0]
// [0  1  0  i]
func (t *FFTTable) fftStage4(data []float64) {
	ns4 := t.n / 4
	pre := data[:ns4]
	pim := data[ns4:]
	for block := 0; block < ns4; block += 4 {
		re := pre[block : block+4]
		im := pim[block : block+4]
		var tmp0, tmp1, tmp2, tmp3 [4]float64
		tmp0[0] = re[0]
		tmp0[1] = re[1]
		tmp0[2] = re[0]
		tmp0[3] = re[1]
		tmp1[0] = re[2]
		tmp1[1] = im[3]
		tmp1[2] = -re[2]
		tmp1[3] = -im[3]
		tmp2[0] = im[0]
		tmp2[1] = im[1]
		tmp2[2] = im[0]
		tmp2[3] = im[1]
		tmp3[0] = im[2]
		tmp3[1] = -re[3]
		tmp3[2] = -im[2]
		tmp3[3] = re[3]
		for i := 0; i < 4; i++ {
			re[i] = tmp0[i] + tmp1[i]
		}
		for i := 0; i < 4; i++ {
			im[i] = tmp2[i] + tmp3[i]
		}
	}
}

// general butterfly stage for halfnn >= 4 (scalar fallback).
func (t *FFTTable) fftGeneralStageScalar(data []float64, halfnn, curTT int) {
	ns4 := t.n / 4
	nn := 2 * halfnn
	pre := data[:ns4]
	pim := data[ns4:]
	for block := 0; block < ns4; block += nn {
		for off := 0; off < halfnn; off += 4 {
			re0 := pre[block+off:]
			im0 := pim[block+off:]
			re1 := pre[block+halfnn+off:]
			im1 := pim[block+halfnn+off:]
			tcs := t.trig[curTT+2*off:]
			tsn := t.trig[curTT+2*off+4:]
			var tmp0, tmp1, tmp2, tmp3 [4]float64
			for i := 0; i < 4; i++ {
				tmp0[i] = re1[i] * tcs[i] // re*cos
				tmp1[i] = re1[i] * tsn[i] // re*sin
				tmp2[i] = im1[i] * tcs[i] // im*cos
				tmp3[i] = im1[i] * tsn[i] // im*sin
			}
			for i := 0; i < 4; i++ {
				tmp0[i] = tmp0[i] - tmp3[i] // re2
				tmp1[i] = tmp1[i] + tmp2[i] // im2
			}
			for i := 0; i < 4; i++ {
				tmp2[i] = re0[i] + tmp0[i] // re + re
				tmp3[i] = im0[i] + tmp1[i] // im + im
				tmp0[i] = re0[i] - tmp0[i] // re - re
				tmp1[i] = im0[i] - tmp1[i] // im - im
			}
			for i := 0; i < 4; i++ {
				re0[i] = tmp2[i]
				im0[i] = tmp3[i]
				re1[i] = tmp0[i]
				im1[i] = tmp1[i]
			}
		}
	}
}

// final twiddles: multiply by omb^j (negative angles).
func (t *FFTTable) fftTwiddleScalar(data []float64, curTT int) {
	ns4 := t.n / 4
	pre := data[:ns4]
	pim := data[ns4:]
	for j := 0; j < ns4; j += 4 {
		r0 := t.trig[curTT+2*j:]
		r1 := r0[4:]
		d0 := pre[j:]
		d1 := pim[j:]
		var tmp0, tmp1, tmp2, tmp3 [4]float64
		for i := 0; i < 4; i++ {
			tmp0[i] = d0[i] * r0[i] // re*cos
			tmp1[i] = d1[i] * r0[i] // im*cos
			tmp2[i] = d0[i] * r1[i] // re*sin
			tmp3[i] = d1[i] * r1[i] // im*sin
		}
		for i := 0; i < 4; i++ {
			d0[i] = tmp0[i] - tmp3[i]
			d1[i] = tmp1[i] + tmp2[i]
		}
	}
}

// ---- reverse stages ------------------------------------------------------

// first twiddles: multiply by omega^j (positive angles). Same shape as
// fftTwiddleScalar with a different table offset.
func (t *IFFTTable) ifftTwiddleScalar(data []float64) {
	ns4 := t.n / 4
	are := data[:ns4]
	aim := data[ns4:]
	for j := 0; j < ns4; j += 4 {
		r0 := t.trig[2*j:]
		r1 := r0[4:]
		d0 := are[j:]
		d1 := aim[j:]
		var tmp0, tmp1, tmp2, tmp3 [4]float64
		for i := 0; i < 4; i++ {
			tmp0[i] = d0[i] * r0[i] // re*cos
			tmp1[i] = d1[i] * r0[i] // im*cos
			tmp2[i] = d0[i] * r1[i] // re*sin
			tmp3[i] = d1[i] * r1[i] // im*sin
		}
		for i := 0; i < 4; i++ {
			d0[i] = tmp0[i] - tmp3[i]
			d1[i] = tmp1[i] + tmp2[i]
		}
	}
}

// general butterfly stage for the reverse transform (scalar fallback);
// nn = 2*halfnn is the current sub-transform size.
func (t *IFFTTable) ifftGeneralStageScalar(data []float64, nn, curTT int) {
	ns4 := t.n / 4
	halfnn := nn / 2
	are := data[:ns4]
	aim := data[ns4:]
	for block := 0; block < ns4; block += nn {
		for off := 0; off < halfnn; off += 4 {
			d00 := are[block+off:]
			d01 := aim[block+off:]
			d10 := are[block+halfnn+off:]
			d11 := aim[block+halfnn+off:]
			var tmp0, tmp1, tmp2, tmp3 [4]float64
			for i := 0; i < 4; i++ {
				tmp0[i] = d00[i] + d10[i] // re + re
				tmp1[i] = d01[i] + d11[i] // im + im
				tmp2[i] = d00[i] - d10[i] // re - re
				tmp3[i] = d01[i] - d11[i] // im - im
			}
			for i := 0; i < 4; i++ {
				d00[i] = tmp0[i]
				d01[i] = tmp1[i]
			}
			r0 := t.trig[curTT+2*off:]
			r1 := r0[4:]
			for i := 0; i < 4; i++ {
				tmp0[i] = tmp2[i] * r0[i] // re*cos
				tmp1[i] = tmp3[i] * r1[i] // im*sin
				d10[i] = tmp0[i] - tmp1[i]
				tmp0[i] = tmp2[i] * r1[i] // re*sin
				tmp1[i] = tmp3[i] * r0[i] // im*cos
				d11[i] = tmp0[i] + tmp1[i]
			}
		}
	}
}

// size 4 (reverse)
func (t *IFFTTable) ifftStage4(data []float64) {
	ns4 := t.n / 4
	are := data[:ns4]
	aim := data[ns4:]
	for block := 0; block < ns4; block += 4 {
		d0 := are[block : block+4]
		d1 := aim[block : block+4]
		var tmp0, tmp1, tmp2, tmp3 [4]float64
		tmp0[0] = d0[0]
		tmp0[1] = d0[1]
		tmp0[2] = d0[0]
		tmp0[3] = -d1[1]
		tmp1[0] = d0[2]
		tmp1[1] = d0[3]
		tmp1[2] = -d0[2]
		tmp1[3] = d1[3]
		tmp2[0] = d1[0]
		tmp2[1] = d1[1]
		tmp2[2] = d1[0]
		tmp2[3] = d0[1]
		tmp3[0] = d1[2]
		tmp3[1] = d1[3]
		tmp3[2] = -d1[2]
		tmp3[3] = -d0[3]
		for i := 0; i < 4; i++ {
			d0[i] = tmp0[i] + tmp1[i]
		}
		for i := 0; i < 4; i++ {
			d1[i] = tmp2[i] + tmp3[i]
		}
	}
}

// size 2 (reverse)
func (t *IFFTTable) ifftStage2(data []float64) {
	ns4 := t.n / 4
	are := data[:ns4]
	aim := data[ns4:]
	for block := 0; block < ns4; block += 4 {
		d0 := are[block : block+4]
		d1 := aim[block : block+4]
		var tmp0, tmp1 [4]float64
		tmp0[0] = d0[0]
		tmp0[1] = d0[0]
		tmp0[2] = d0[2]
		tmp0[3] = d0[2]
		tmp1[0] = d0[1]
		tmp1[1] = -d0[1]
		tmp1[2] = d0[3]
		tmp1[3] = -d0[3]
		for i := 0; i < 4; i++ {
			d0[i] = tmp0[i] + tmp1[i]
		}
		tmp0[0] = d1[0]
		tmp0[1] = d1[0]
		tmp0[2] = d1[2]
		tmp0[3] = d1[2]
		tmp1[0] = d1[1]
		tmp1[1] = -d1[1]
		tmp1[2] = d1[3]
		tmp1[3] = -d1[3]
		for i := 0; i < 4; i++ {
			d1[i] = tmp0[i] + tmp1[i]
		}
	}
}
