"""Uji simpanan katalog: cepat, tetapi tidak boleh pernah menyajikan yang salah.

Cache adalah tempat lahirnya cacat yang paling tidak enak: ia tidak membuat uji
gagal, ia membuat jawaban usang. Dua di antaranya serius di layanan ini -
materi yang baru diunggah tidak muncul, dan daftar satu kampus tersaji ke
kampus lain. Keduanya diuji di sini secara langsung, bukan disimpulkan.

Klien Qdrant dipalsukan dan `scroll` dihitung panggilannya: itulah satu-satunya
cara membuktikan simpanan benar-benar dipakai. Mengukur waktu tidak dapat
dipercaya pada mesin yang sedang sibuk.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.schemas import Chunk, ElementType
from src.storage import catalog_cache
from src.storage.qdrant_store import QdrantStore

TENANT_A = "kampus-a"
TENANT_B = "kampus-b"


def _titik(**payload) -> SimpleNamespace:
    return SimpleNamespace(payload=payload)


def _tenant_dari(qfilter) -> str | None:
    """Ambil tenant_id dari filter yang dibangun `_build_filter`.

    Dibaca dari filternya, bukan dari urutan panggilan: uji isolasi harus
    membuktikan bahwa tiap kampus menerima datanya SENDIRI, dan itu hanya
    bermakna bila data palsunya benar-benar dipilih berdasarkan saringannya.
    """
    for syarat in getattr(qfilter, "must", None) or []:
        if getattr(syarat, "key", None) == "tenant_id":
            cocok = getattr(syarat, "match", None)
            if cocok is not None:
                return getattr(cocok, "value", None)
    return None


def _store(data: dict[str, list]) -> tuple[MagicMock, QdrantStore]:
    """Store dengan klien palsu yang menjawab sesuai tenant di filternya."""
    client = MagicMock()

    def scroll(**kw):
        tenant = _tenant_dari(kw.get("scroll_filter"))
        return list(data.get(tenant or "", [])), None

    client.scroll.side_effect = scroll
    return client, QdrantStore(
        client=client, collection="uji", enable_sparse=False,
    )


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="c1",
        text="isi",
        parent_element_id="e1",
        element_type=ElementType.TEXT,
        source_file="bab1.pdf",
        page_number=1,
        dense_embedding=[0.1] * 1024,
    )


class TestSimpananDipakai:
    @pytest.mark.asyncio
    async def test_panggilan_kedua_tidak_menyentuh_penyimpanan(self) -> None:
        client, store = _store({TENANT_A: [_titik(course_id="sbd", course_name="SBD")]})

        pertama = await store.list_courses(tenant_id=TENANT_A)
        sesudah_satu = client.scroll.call_count
        kedua = await store.list_courses(tenant_id=TENANT_A)

        assert pertama == kedua == [{"course_id": "sbd", "course_name": "SBD"}]
        assert sesudah_satu == 1
        assert client.scroll.call_count == 1, (
            "panggilan kedua masih memindai penyimpanan; simpanan tidak terpakai"
        )

    @pytest.mark.asyncio
    async def test_minggu_terbalik_bukan_dua_entri(self) -> None:
        """[3, 1] dan [1, 3] adalah permintaan yang sama persis."""
        client, store = _store({
            TENANT_A: [_titik(source_file="bab1.pdf", content_id="x", week=1)],
        })

        await store.list_materials_for_weeks("sbd", [1, 3], tenant_id=TENANT_A)
        await store.list_materials_for_weeks("sbd", [3, 1], tenant_id=TENANT_A)

        assert client.scroll.call_count == 1


class TestSimpananTidakMenyesatkan:
    @pytest.mark.asyncio
    async def test_unggahan_baru_langsung_terlihat(self) -> None:
        """Inti perbaikan ini: cepat tidak boleh berarti basi.

        Dosen yang mengunggah materi lalu tidak melihatnya di daftar akan
        mengunggah ulang, dan itu menghasilkan materi kembar - bukan sekadar
        tampilan yang keliru.
        """
        client, store = _store({TENANT_A: [_titik(course_id="sbd", course_name="SBD")]})

        await store.list_courses(tenant_id=TENANT_A)
        assert client.scroll.call_count == 1

        await store.upsert_chunks([_chunk()], tenant_id=TENANT_A)
        await store.list_courses(tenant_id=TENANT_A)

        assert client.scroll.call_count == 2, (
            "daftar masih dijawab dari simpanan setelah ada penulisan"
        )

    @pytest.mark.asyncio
    async def test_penghapusan_juga_membatalkan(self) -> None:
        client, store = _store({TENANT_A: [_titik(course_id="sbd", course_name="SBD")]})

        await store.list_courses(tenant_id=TENANT_A)
        await store.delete_by_source("bab1.pdf", tenant_id=TENANT_A)
        await store.list_courses(tenant_id=TENANT_A)

        assert client.scroll.call_count == 2

    @pytest.mark.asyncio
    async def test_perubahan_dari_pekerja_lain_terbaca(self) -> None:
        """Produksi berjalan 4 pekerja uvicorn; tiap pekerja proses terpisah.

        Unggahan yang dilayani pekerja lain hanya terlihat di sini bila
        penandanya dibaca dari disk, bukan disimpan di memori proses ini.
        Berkas penandanya ditulis langsung di sini - itulah yang dilakukan
        pekerja lain dari sudut pandang proses ini.
        """
        client, store = _store({TENANT_A: [_titik(course_id="sbd", course_name="SBD")]})

        await store.list_courses(tenant_id=TENANT_A)
        assert client.scroll.call_count == 1

        catalog_cache.DIR_PENANDA.mkdir(parents=True, exist_ok=True)
        (catalog_cache.DIR_PENANDA / f"{TENANT_A}.stamp").write_text(
            "pekerja-lain", encoding="utf-8",
        )
        await store.list_courses(tenant_id=TENANT_A)

        assert client.scroll.call_count == 2

    @pytest.mark.asyncio
    async def test_daftar_yang_diubah_pemanggil_tidak_merusak_simpanan(self) -> None:
        """Yang dikembalikan harus salinan, bukan isi simpanan itu sendiri."""
        _, store = _store({
            TENANT_A: [
                _titik(course_id="sbd", course_name="SBD"),
                _titik(course_id="kka", course_name="KKA"),
            ],
        })

        pertama = await store.list_courses(tenant_id=TENANT_A)
        pertama.clear()
        pertama_dict = await store.list_courses(tenant_id=TENANT_A)
        assert len(pertama_dict) == 2, "daftar di simpanan ikut terpotong"

        pertama_dict[0]["course_name"] = "DIUBAH"
        kedua = await store.list_courses(tenant_id=TENANT_A)
        assert all(c["course_name"] != "DIUBAH" for c in kedua), (
            "dict di dalam simpanan ikut berubah"
        )


class TestIsolasiKampus:
    @pytest.mark.asyncio
    async def test_kampus_lain_tidak_menerima_daftar_kampus_ini(self) -> None:
        """Kunci simpanan menyertakan tenant; tanpa itu ini kebocoran data."""
        client, store = _store({
            TENANT_A: [_titik(course_id="sbd", course_name="SBD")],
            TENANT_B: [_titik(course_id="kalkulus", course_name="Kalkulus")],
        })

        milik_a = await store.list_courses(tenant_id=TENANT_A)
        milik_b = await store.list_courses(tenant_id=TENANT_B)

        assert milik_a == [{"course_id": "sbd", "course_name": "SBD"}]
        assert milik_b == [{"course_id": "kalkulus", "course_name": "Kalkulus"}]
        assert client.scroll.call_count == 2, (
            "kampus kedua dijawab dari simpanan kampus pertama"
        )

    @pytest.mark.asyncio
    async def test_unggahan_satu_kampus_tidak_membatalkan_kampus_lain(self) -> None:
        """Penanda per kampus, bukan global.

        Kalau satu penanda dipakai bersama, kampus yang ramai mengunggah akan
        terus-menerus membatalkan simpanan kampus lain, dan perbaikan ini tidak
        berguna di layanan yang memang multi-kampus.
        """
        client, store = _store({
            TENANT_A: [_titik(course_id="sbd", course_name="SBD")],
            TENANT_B: [_titik(course_id="kalkulus", course_name="Kalkulus")],
        })

        await store.list_courses(tenant_id=TENANT_A)
        await store.list_courses(tenant_id=TENANT_B)
        assert client.scroll.call_count == 2

        await store.upsert_chunks([_chunk()], tenant_id=TENANT_B)

        await store.list_courses(tenant_id=TENANT_A)
        assert client.scroll.call_count == 2, "simpanan kampus A ikut terbuang"

        await store.list_courses(tenant_id=TENANT_B)
        assert client.scroll.call_count == 3


class TestGagalDenganBenar:
    @pytest.mark.asyncio
    async def test_penanda_tak_tertulis_mematikan_simpanan(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Kalau penanda tidak dapat dipakai, memindai ulang lebih baik daripada basi.

        Kegagalannya dibuat sungguhan, bukan ditiru dengan mock: direktori
        penanda diarahkan ke dalam sebuah BERKAS, sehingga baik `mkdir` maupun
        `read_text` melempar OSError seperti pada disk yang bermasalah.
        """
        penghalang = tmp_path / "bukan-direktori"
        penghalang.write_text("saya berkas", encoding="utf-8")
        monkeypatch.setattr(catalog_cache, "DIR_PENANDA", penghalang / "catalog")

        client, store = _store({TENANT_A: [_titik(course_id="sbd", course_name="SBD")]})

        hasil = [await store.list_courses(tenant_id=TENANT_A) for _ in range(3)]

        assert all(h == [{"course_id": "sbd", "course_name": "SBD"}] for h in hasil), (
            "jawaban harus tetap benar meski simpanan mati"
        )
        assert client.scroll.call_count == 3, (
            "menyimpan tanpa penanda berarti menyajikan katalog basi tanpa "
            "batas waktu"
        )
