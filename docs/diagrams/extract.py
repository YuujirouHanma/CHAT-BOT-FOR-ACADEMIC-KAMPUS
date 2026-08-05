"""Ekstrak blok ```mermaid dari ARCHITECTURE_DIAGRAM.md menjadi berkas .mmd terpisah.

Nama berkas diambil dari heading "## Fig. N - Judul" yang mendahului tiap blok,
sehingga urutan berkas selalu cocok dengan penomoran gambar di dokumen.

    python docs/diagrams/extract.py
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "ARCHITECTURE_DIAGRAM.md"
OUT = Path(__file__).resolve().parent / "src"

# Heading gambar, mis. "## Fig. 4 — Mekanisme Hybrid Retrieval dengan Fusi RRF"
_HEADING = re.compile(r"^##\s*Fig\.\s*(\d+)\s*[—–-]\s*(.+?)\s*$", re.M)
_BLOCK = re.compile(r"```mermaid\n(.*?)```", re.S)


def slugify(text: str) -> str:
    """'Fusi RRF & Retrieval' -> 'fusi-rrf-retrieval' (ASCII, aman untuk nama berkas)."""
    ascii_text = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    )
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return re.sub(r"-{2,}", "-", slug)


def main() -> None:
    source = DOC.read_text(encoding="utf-8")

    # Pasangkan tiap blok dengan heading Fig. terdekat SEBELUMNYA.
    headings = [(m.start(), m.group(1), m.group(2)) for m in _HEADING.finditer(source)]
    OUT.mkdir(parents=True, exist_ok=True)

    # Sebuah gambar boleh terdiri atas beberapa panel (mis. Fig. 2a dan 2b).
    # Panel kedua dan seterusnya mendapat sufiks huruf agar tidak saling menimpa.
    seen: dict[str, int] = {}

    written = 0
    for block in _BLOCK.finditer(source):
        preceding = [h for h in headings if h[0] < block.start()]
        if preceding:
            _, number, title = preceding[-1]
            base = f"fig{int(number):02d}-{slugify(title)}"
        else:
            base = f"extra{written + 1:02d}"

        seen[base] = seen.get(base, 0) + 1
        if seen[base] == 1:
            name = base
        else:
            # Panel ke-2 jadi '...-b'; panel pertama diganti nama menjadi '...-a'.
            suffix = chr(ord("a") + seen[base] - 1)
            name = f"{base}-{suffix}"
            first = OUT / f"{base}.mmd"
            if first.exists():
                first.rename(OUT / f"{base}-a.mmd")
                print(f"  {base}.mmd -> {base}-a.mmd")

        target = OUT / f"{name}.mmd"
        target.write_text(block.group(1).rstrip() + "\n", encoding="utf-8")
        print(f"  {target.name}")
        written += 1

    print(f"\n{written} berkas .mmd ditulis ke {OUT}")


if __name__ == "__main__":
    main()
