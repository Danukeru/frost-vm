/*
 *
 * Maps a normal C program onto the machine's two encrypted memories:
 * .text at address 0 (instruction slot = vaddr/4) and .data/.bss at
 * 0x1000, which is congruent to 0 modulo every power-of-two data-memory
 * size the VM uses, so any data vaddr lands at mem[vaddr & (nMem-1)].
 * The whole contract between compiler output and loader lives in these
 * few lines; everything else is discarded metadata.
 *
 * Original header follows.
 *
 * Link script for RV64 programs on the frost-vm oblivious VM.
 *
 * The VM has separate encrypted program and data memories. .text is
 * placed at byte address 0 (instruction slot vaddr/4); .data/.bss at
 * 0x1000 — the VM masks data addresses to its memory size, and 0x1000
 * is congruent to 0 modulo every power-of-two memory size we use, so
 * data vaddrs map to mem[vaddr & (nMem-1)].
 */
ENTRY(_start)
SECTIONS
{
  . = 0x0;
  .text : {
    *(.text .text.*)
  }
  . = 0x1000;
  .data : {
    *(.data .data.*)
    *(.sdata .sdata.*)
  }
  .bss (NOLOAD) : {
    *(.bss .bss.*)
    *(.sbss .sbss.*)
    *(COMMON)
  }
  /DISCARD/ : {
    *(.comment)
    *(.riscv.attributes)
    *(.note*)
    *(.eh_frame*)
  }
}
