// Package vm loads standard RV64I ELF segments into encrypted machine memories.
//
// LoadELF validates ELF class, architecture, entry point, and machine geometry.
// It sizes registers from format-aware field use, avoiding immediate-bit aliases.
// Text words map to instruction slots; data bytes map to encrypted RAM addresses.
// Validation precedes state loading, and instruction encodings are preserved.
// Execution afterward uses the normal decode-once masked-OR cycle; x0 and halt
// behavior remain machine invariants rather than loader-side plaintext decisions.
package vm

import (
	"debug/elf"
	"encoding/binary"
	"fmt"
)

// regsUsed returns the register numbers an RV64 instruction reads or
// writes, per its format (R/I/S/U/J/B). Opcodes outside the base ISA
// (and FENCE) contribute none; the machine treats them as no-ops.
func regsUsed(i uint32) []uint32 {
	rd := (i >> 7) & 0x1F
	rs1 := (i >> 15) & 0x1F
	rs2 := (i >> 20) & 0x1F
	switch i & 0x7F {
	case 0x33: // OP (register ALU)
		return []uint32{rd, rs1, rs2}
	case 0x13, 0x03, 0x67, 0x73: // OP-IMM, LOAD, JALR, SYSTEM
		return []uint32{rd, rs1}
	case 0x23: // STORE
		return []uint32{rs1, rs2}
	case 0x37, 0x17, 0x6F: // LUI, AUIPC, JAL
		return []uint32{rd}
	case 0x63: // BRANCH
		return []uint32{rs1, rs2}
	}
	return nil
}

// LoadELF loads an RV64I ELF executable into the machine: executable
// PT_LOAD segments go to program memory (byte vaddr -> instruction slot
// vaddr/4), writable/read-only data segments to data memory (vaddr is
// taken modulo nMem, exactly like the machine's address masking).
//
// The program must fit: text size <= 4*nProg bytes, data vaddrs within
// nMem bytes after masking. The entry point must be 0 (the machine's PC
// starts there).
func (rv *RVMachine) LoadELF(path string) error {
	f, err := elf.Open(path)
	if err != nil {
		return fmt.Errorf("vm: open elf: %w", err)
	}
	defer f.Close()
	if f.Machine != elf.EM_RISCV || f.Class != elf.ELFCLASS64 {
		return fmt.Errorf("vm: %s: not an RV64 ELF (machine=%v class=%v)", path, f.Machine, f.Class)
	}
	if len(f.Progs) == 0 {
		return fmt.Errorf("vm: %s: no program headers", path)
	}

	// collect segments first, validating geometry before touching state
	type seg struct {
		x, w   bool
		vaddr  uint64
		data   []byte
		memSz  uint64
	}
	var segs []seg
	entry := f.Entry
	if entry != 0 {
		return fmt.Errorf("vm: entry point %#x != 0 (machine PC starts at 0)", entry)
	}
	for _, p := range f.Progs {
		if p.Type != elf.PT_LOAD {
			continue
		}
		data := make([]byte, p.Filesz)
		if p.Filesz > 0 {
			if _, err := p.ReadAt(data, 0); err != nil {
				return fmt.Errorf("vm: read segment at %#x: %w", p.Vaddr, err)
			}
		} // .bss-only segment: Filesz=0, stays zero in memory
		s := seg{
			x:     p.Flags&elf.PF_X != 0,
			w:     p.Flags&elf.PF_W != 0,
			vaddr: p.Vaddr,
			data:  data,
			memSz: p.Memsz,
		}
		if s.x {
			if s.vaddr%4 != 0 || s.vaddr+s.memSz > uint64(4*rv.nProg) {
				return fmt.Errorf("vm: text [%#x,%#x) does not fit %d instruction slots",
					s.vaddr, s.vaddr+s.memSz, rv.nProg)
			}
		} else {
			if s.vaddr%uint64(rv.nMem)+s.memSz > uint64(rv.nMem) {
				return fmt.Errorf("vm: data [%#x,%#x) does not fit %d bytes of data memory",
					s.vaddr, s.vaddr+s.memSz, rv.nMem)
			}
		}
		segs = append(segs, s)
	}
	if len(segs) == 0 {
		return fmt.Errorf("vm: %s: no PT_LOAD segments", path)
	}

	// scan text for the highest register actually used, to size the
	// register file honestly. Only fields the instruction format really
	// uses count: immediate bits overlap rd/rs1/rs2 positions in other
	// formats, so a naive field scan would report phantom registers.
	maxReg := 0
	for _, s := range segs {
		if !s.x {
			continue
		}
		for off := 0; off+4 <= len(s.data); off += 4 {
			i := binary.LittleEndian.Uint32(s.data[off:])
			for _, r := range regsUsed(i) {
				if int(r) > maxReg {
					maxReg = int(r)
				}
			}
		}
	}
	if maxReg+1 > rv.nRegs {
		return fmt.Errorf("vm: program uses register x%d but the machine has only %d registers (rebuild with larger nRegs)", maxReg, rv.nRegs)
	}

	// load
	rv.insts = make([]*Word, rv.nProg)
	for i := range rv.insts {
		rv.insts[i] = rv.EncryptWord(0, 32)
	}
	for _, s := range segs {
		if s.x {
			for off := 0; off+4 <= len(s.data); off += 4 {
				insn := binary.LittleEndian.Uint32(s.data[off:])
				slot := int(s.vaddr)/4 + off/4
				if insn != 0 {
					rv.insts[slot] = rv.EncryptWord(uint64(insn), 32)
				}
			}
		} else {
			for i := uint64(0); i < s.memSz; i++ {
				var b byte
				if i < uint64(len(s.data)) {
					b = s.data[i]
				} // .bss (i >= filesz) stays zero, like Reset()
				if b != 0 {
					rv.mem[(s.vaddr+i)%uint64(rv.nMem)] = rv.EncryptWord(uint64(b), 8)
				}
			}
		}
	}
	return nil
}
