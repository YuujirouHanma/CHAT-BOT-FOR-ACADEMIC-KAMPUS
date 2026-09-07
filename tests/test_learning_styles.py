"""Tests for gaya belajar dan pembentukan notebook.

Gaya belajar mengganti system prompt yang mengatur CARA LLM menjawab. Kalau
suffix-nya tidak benar-benar masuk ke prompt, fiturnya diam-diam tidak berefek
sama sekali — perubahan hanya terasa pada nada, bukan pada isi. Karena itu
penggabungan prompt diuji langsung.
"""
from __future__ import annotations

import json

import pytest

from src import learning_styles
from src.generation.notebook import build_notebook, has_code, safe_filename
from src.generation.prompts import build_system_prompt


class TestRegistry:
    def test_all_styles_have_complete_metadata(self) -> None:
        for s in learning_styles.all_styles():
            assert s.key and s.label and s.description
            assert s.system_suffix.strip(), f"{s.key} tanpa instruksi prompt"
            assert s.greeting.strip(), f"{s.key} tanpa sambutan"

    def test_keys_are_unique(self) -> None:
        keys = [s.key for s in learning_styles.all_styles()]
        assert len(keys) == len(set(keys))

    def test_default_style_exists(self) -> None:
        assert learning_styles.get(learning_styles.DEFAULT_STYLE) is not None

    def test_unknown_key_returns_none_but_resolve_falls_back(self) -> None:
        assert learning_styles.get("tidak-ada") is None
        assert learning_styles.resolve("tidak-ada").key == learning_styles.DEFAULT_STYLE
        assert learning_styles.resolve(None).key == learning_styles.DEFAULT_STYLE

    def test_only_praktik_produces_notebook(self) -> None:
        penghasil = [s.key for s in learning_styles.all_styles() if s.produces_notebook]
        assert penghasil == ["praktik"]


class TestMatch:
    @pytest.mark.parametrize(
        ("teks", "harap"),
        [
            ("visual", "visual"),
            ("pakai diagram dong", "visual"),
            ("mau lihat bagan", "visual"),
            ("kasih contoh kode", "praktik"),
            ("mau notebook", "praktik"),
            ("ringkas saja", "ringkas"),
            ("poin-poin", "ringkas"),
            ("tanya balik saja", "sokratik"),
            ("jelaskan pelan-pelan", "naratif"),
            ("pakai analogi", "naratif"),
        ],
    )
    def test_recognises_style_from_free_text(self, teks: str, harap: str) -> None:
        assert learning_styles.match(teks) == harap

    @pytest.mark.parametrize("teks", ["apa itu normalisasi?", "sbd minggu 3", ""])
    def test_returns_none_when_no_style_mentioned(self, teks: str) -> None:
        assert learning_styles.match(teks) is None


class TestSystemPrompt:
    def test_style_suffix_actually_enters_the_prompt(self) -> None:
        """Kalau ini gagal, memilih gaya tidak berefek apa pun pada jawaban."""
        for s in learning_styles.all_styles():
            prompt = build_system_prompt(style=s.key)
            assert s.system_suffix.strip()[:40] in prompt

    def test_style_and_level_combine(self) -> None:
        prompt = build_system_prompt(level="sederhana", style="visual")
        assert "mermaid" in prompt.lower()          # dari gaya
        assert "SANGAT SEDERHANA" in prompt         # dari level

    def test_unknown_style_leaves_prompt_unchanged(self) -> None:
        assert build_system_prompt(style="ngawur") == build_system_prompt()

    def test_visual_style_demands_a_diagram(self) -> None:
        assert "mermaid" in build_system_prompt(style="visual").lower()

    def test_socratic_style_forbids_immediate_answer(self) -> None:
        p = build_system_prompt(style="sokratik").lower()
        assert "jangan langsung" in p


class TestNotebook:
    JAWABAN = (
        "Berikut cara menghitung total belanja.\n\n"
        "```python\n"
        "harga = [1000, 2000]\n"
        "print(sum(harga))\n"
        "```\n\n"
        "Keluarannya 3000."
    )

    def test_detects_runnable_code(self) -> None:
        assert has_code(self.JAWABAN) is True
        assert has_code("Tidak ada kode di sini.") is False

    def test_no_notebook_without_code(self) -> None:
        assert build_notebook("Penjelasan biasa tanpa kode.") is None

    def test_notebook_is_valid_json_and_shape(self) -> None:
        nb = build_notebook(self.JAWABAN, title="Belanja")
        assert nb is not None
        d = json.loads(nb)
        assert d["nbformat"] == 4
        assert d["metadata"]["kernelspec"]["name"] == "python3"

    def test_prose_becomes_markdown_and_code_becomes_code(self) -> None:
        d = json.loads(build_notebook(self.JAWABAN) or "{}")
        jenis = [c["cell_type"] for c in d["cells"]]
        assert "code" in jenis and "markdown" in jenis
        kode = [c for c in d["cells"] if c["cell_type"] == "code"]
        assert "".join(kode[0]["source"]).startswith("harga = ")

    def test_non_python_block_kept_as_markdown(self) -> None:
        """Blok SQL tidak boleh jadi sel kode Python, tapi isinya jangan hilang."""
        d = json.loads(build_notebook(
            "Contoh:\n\n```sql\nSELECT 1;\n```\n\n```python\nx = 1\n```"
        ) or "{}")
        semua = "".join("".join(c["source"]) for c in d["cells"])
        assert "SELECT 1;" in semua
        kode = [c for c in d["cells"] if c["cell_type"] == "code"]
        assert len(kode) == 1                     # hanya blok python

    @pytest.mark.parametrize(
        ("nama", "harap"),
        [
            ("Materi SBD TM9(Materi).pptx", "materi-sbd-tm9materipptx.ipynb"),
            ("", "catatan.ipynb"),
        ],
    )
    def test_safe_filename(self, nama: str, harap: str) -> None:
        assert safe_filename(nama) == harap


class TestRegresiNamaMataKuliah:
    """Nama mata kuliah tidak boleh diam-diam menetapkan gaya belajar.

    Ditemukan saat uji ujung-ke-ujung: pencocokan dulu memakai substring, dan
    isyarat "dasar" ada di dalam "Dasar Pemrograman". Akibatnya sekadar MEMILIH
    mata kuliah sudah menetapkan gaya, sehingga langkah "mau dijelaskan dengan
    cara apa?" terlewat tanpa ada yang menyadarinya — mematikan rotasi gaya
    yang justru menjadi inti rancangan penelitiannya.
    """

    @pytest.mark.parametrize(
        "nama",
        [
            "Dasar Pemrograman (RKA)",
            "Dasar Pemrograman (IF)",
            "Sistem Basis Data",
            "Konsep Kecerdasan Artifisial",
            "Rekayasa Sistem Berbasis Pengetahuan",
            "Struktur Data",
            "Minggu 2",
            "Branching and Iteration - MIT.pdf",
            "apa itu perulangan",
            "jelaskan program ini",
            "berikan contoh soal",
        ],
    )
    def test_pilihan_navigasi_tidak_menetapkan_gaya(self, nama: str) -> None:
        assert learning_styles.match(nama) is None

    @pytest.mark.parametrize(
        ("teks", "harapan"),
        [
            ("pakai diagram dong", "visual"),
            ("mau lewat kode", "praktik"),
            ("ringkas aja", "ringkas"),
            ("jelaskan pelan-pelan", "naratif"),
            ("Lewat diagram", "visual"),
            ("Lewat contoh & kode", "praktik"),
            ("Poin-poin ringkas", "ringkas"),
        ],
    )
    def test_permintaan_gaya_sungguhan_tetap_dikenali(
        self, teks: str, harapan: str,
    ) -> None:
        assert learning_styles.match(teks) == harapan


class TestLabelTombolBukanPertanyaan:
    """Label tombol gaya harus dikenali sebagai PILIHAN, bukan pertanyaan.

    Ditemukan saat uji coba manual: dua dari lima label — "Lewat contoh & kode
    — Banyak contoh nyata…" dan "Poin-poin ringkas — Padat dan langsung ke
    inti…" — cukup panjang dan berisi banyak kata isi sehingga lolos sebagai
    pertanyaan, lalu dijawab sungguhan. Mahasiswa menunggu dua menit untuk
    jawaban atas teks tombol yang baru saja ia tekan, dan satu panggilan LLM
    terbuang.
    """

    def test_setiap_label_dikenali_sebagai_pilihan(self) -> None:
        for spec in learning_styles.all_styles():
            label = learning_styles.choice_label(spec)
            assert learning_styles.is_choice_label(label), spec.key

    def test_label_pendek_dan_key_juga_dikenali(self) -> None:
        # Klien boleh mengirim balik label saja, atau key-nya.
        for spec in learning_styles.all_styles():
            assert learning_styles.is_choice_label(spec.label)
            assert learning_styles.is_choice_label(spec.key)

    @pytest.mark.parametrize(
        "teks",
        [
            "apa itu perulangan while",
            "jelaskan perbedaan for dan while",
            "bagaimana cara kerja break",
            "kenapa indentasi penting di Python",
        ],
    )
    def test_pertanyaan_sungguhan_bukan_label(self, teks: str) -> None:
        assert not learning_styles.is_choice_label(teks)

    def test_label_dirakit_dari_satu_sumber(self) -> None:
        # Tombol dibuat pipeline dan dikenali di sini; kalau bentuknya dirakit
        # di dua tempat, keduanya bisa menyimpang tanpa ada yang menyadarinya.
        spec = learning_styles.resolve("ringkas")
        assert learning_styles.choice_label(spec) == f"{spec.label} — {spec.description}"
