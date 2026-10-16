//go:build !amd64

// This file keeps the kernel interface available on non-amd64 targets while
// the FFT package uses its portable scalar implementation. The AVX entry
// points are inert build stubs, and both hasAVX and useAVX disable dispatch so
// no stub is called by the transform pipeline. The shared signatures preserve
// source portability without changing the SoA buffer contract.
package fft

// Non-amd64 builds provide no-op kernel entry points and report AVX disabled;
// useAVX remains false so the scalar FFT and Fourier operations handle all work.
func hasAVX() bool { return false }

func fftStageAVX(data *float64, trig *float64, ns4, halfnn, ttOff int)  {}
func twiddleMulAVX(data *float64, trig *float64, ns4, ttOff int)        {}
func ifftStageAVX(data *float64, trig *float64, ns4, halfnn, ttOff int) {}
func fourierMulAVX(r *float64, a *float64, b *float64, n int)           {}
func fourierAddMulAVX(r *float64, a *float64, b *float64, n int)        {}

var useAVX = false
