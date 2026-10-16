#!/bin/sh
# The toolchain entry point: compiles every .c in this directory to a
# bare-metal RV64I ELF with clang+lld, using the exact flag set the
# machine's loader expects (base ISA only, absolute medlow addressing,
# no relaxation, our linker script). It then disassembles each output so
# codegen changes are visible at build time rather than as wrong values
# hours into an oblivious run. See docs/toolchain.md for why each flag
# is required.
#
# Build RV64I programs for the frost-vm oblivious FHE VM with clang.
# The VM implements the RV64I base integer ISA only (no M/A/F/D), so
# -march=rv64i; medlow + no-relax keeps addressing absolute (LUI-based),
# which the ELF loader maps directly. lld is required (the host GNU ld
# has no riscv emulation).
set -e
cd "$(dirname "$0")"
CLANG=${CLANG:-clang}
OBJDUMP=${OBJDUMP:-llvm-objdump}
for src in ./*.c; do
	out="${src%.c}.elf"
	"$CLANG" --target=riscv64-unknown-elf \
		-march=rv64i -mabi=lp64 -mcmodel=medlow -mno-relax \
		-nostdlib -nostartfiles -O2 \
		-fuse-ld=lld -Wl,--no-relax \
		-T rv64vm.ld \
		-o "$out" "$src"
	echo "built $out"
	"$OBJDUMP" -d "$out" | sed -n '/<_start>:/,$p'
done
