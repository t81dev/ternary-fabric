#!/usr/bin/env python3
import re
from pathlib import Path
import sys

HEADER = Path("include/tfmbs_dma_regs.h")
DOC = Path("docs/04_MEMORY_MAP.md")

macro_re = re.compile(r"#define\s+TFMBS_REG_([A-Z0-9_]+)\s+(0x[0-9A-Fa-f]+)")
doc_re = re.compile(r"\|\s*0x([0-9A-Fa-f]+)\s*\|\s*`?([A-Za-z0-9_]+)`?\s*\|")

def parse_header():
    if not HEADER.exists():
        raise SystemExit(f"{HEADER} does not exist")
    mapping = {}
    for line in HEADER.read_text().splitlines():
        m = macro_re.match(line)
        if m:
            name, offset = m.groups()
            mapping[name] = int(offset, 16)
    return mapping

def parse_doc():
    if not DOC.exists():
        raise SystemExit(f"{DOC} does not exist")
    mapping = {}
    for line in DOC.read_text().splitlines():
        m = doc_re.match(line)
        if m:
            offset, name = m.groups()
            mapping[name.upper()] = int(offset, 16)
    return mapping

def main():
    header = parse_header()
    doc = parse_doc()

    mismatches = False

    for name, header_offset in header.items():
        doc_offset = doc.get(name)
        if doc_offset is None:
            print(f"[WARNING] {name} defined in header but missing from docs")
            mismatches = True
            continue
        if doc_offset != header_offset:
            print(f"[ERROR] {name}: header=0x{header_offset:02X} doc=0x{doc_offset:02X}")
            mismatches = True

    for name, doc_offset in doc.items():
        if name not in header:
            print(f"[WARNING] {name} mentioned in docs but missing from header")
            mismatches = True

    if mismatches:
        raise SystemExit("Register map validation failed")
    print("Register map validation succeeded")

if __name__ == "__main__":
    main()
