"""Pemeriksaan lingkungan runtime, bukan logika aplikasi.

Ada satu kelas kerusakan yang tidak bisa ditangkap uji unit biasa: torch dan
torchvision yang berasal dari indeks berbeda. Keduanya terpasang, keduanya bisa
diimpor sendiri-sendiri, tapi operator asli torchvision tidak pernah terdaftar
di torch. Akibatnya transformers gagal mengimpor donut dan SELURUH parsing PDF
mati — sementara pptx dan docx tetap normal, sehingga kerusakannya lolos dari
pengujian sekilas. Persis itu yang terjadi setelah torch dipindah ke varian CPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def test_torch_dan_torchvision_dari_build_yang_sama() -> None:
    """Keduanya harus sama-sama '+cpu' atau sama-sama polos, tidak campur."""
    import importlib.metadata as metadata

    varian_torch = "+cpu" in metadata.version("torch")
    varian_vision = "+cpu" in metadata.version("torchvision")

    assert varian_torch == varian_vision, (
        f"torch={metadata.version('torch')} tidak sepadan dengan "
        f"torchvision={metadata.version('torchvision')}; pasang keduanya dari "
        "indeks yang sama di Dockerfile."
    )


def test_operator_torchvision_terdaftar_di_torch() -> None:
    """nms adalah operator yang pertama gagal saat ABI-nya tidak cocok."""
    assert hasattr(torch.ops.torchvision, "nms")


def test_parser_pdf_bisa_diimpor() -> None:
    """Jalur impor sebenarnya yang mematikan parsing PDF saat ABI tidak cocok."""
    pytest.importorskip("transformers")
    from transformers.models.donut import processing_donut  # noqa: F401
