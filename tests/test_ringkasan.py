"""Ringkasan riwayat chat untuk dosen.

Dua kelompok uji dengan alasan yang berbeda.

`TestPenandaPenolakan` menjaga sambungan yang paling rapuh di seluruh fitur ini:
`_PENANDA_TAK_TERJAWAB` berisi potongan kalimat yang disusun di tempat LAIN
(src/pipeline.py). Kalau kalimat di sana diperhalus, daftar penanda tetap lulus
semua uji yang memakai teks salinan tangan — dan angka "pertanyaan tidak
terjawab" diam-diam menjadi nol tanpa satu pun uji merah. Karena itu uji di
bawah memanggil PEMBUAT PESANNYA, bukan menyalin teksnya.

Sisanya menjaga angka-angkanya: satu kesalahan hitung di sini menghasilkan
laporan yang terbaca masuk akal dan salah, dan dosen tidak punya cara
memeriksanya.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.hitl.ringkasan import (
    _kunci_pertanyaan,
    _tak_terjawab,
    hitung_fakta,
    susun_bahan,
)
from src.pipeline import RAGPipeline
from tests.conftest import TEST_TENANT_ID as TENANT


def _rec(
    *, question: str = "apa itu basis data", answer: str = "Jawaban.",
    user_id: int | None = 1, timestamp: str = "2026-10-20T10:00:00",
    topik: str | None = None, tipe: str | None = None,
    sumber: list[str] | None = None,
) -> dict:
    """Satu baris log seperti yang benar-benar ditulis `log_interaction`."""
    return {
        "question": question,
        "answer": answer,
        "user_id": user_id,
        "timestamp": timestamp,
        "decomposition": {"topik_utama": topik, "tipe_pertanyaan": tipe},
        "sources": [{"source_file": s} for s in (sumber or [])],
    }


class TestPenandaPenolakan:
    """Penanda diuji terhadap pesan yang SUNGGUHAN dibuat, bukan teks salinan."""

    @pytest.fixture
    def pipa(self) -> RAGPipeline:
        """Pipeline tanpa memuat model: hanya `_store` yang dipakai di sini."""
        p = RAGPipeline.__new__(RAGPipeline)
        store = AsyncMock()
        store.list_weeks = AsyncMock(return_value=[1, 2, 3])
        p._store = store
        return p

    async def test_pesan_tanpa_mata_kuliah_dikenali(self, pipa) -> None:
        pesan = await pipa._pesan_tak_terjawab(
            course_id=None, weeks=None, tenant_id=TENANT,
        )

        assert _tak_terjawab(pesan), f"penanda tidak cocok dengan: {pesan!r}"

    async def test_pesan_dengan_daftar_minggu_dikenali(self, pipa) -> None:
        pesan = await pipa._pesan_tak_terjawab(
            course_id="sbd", weeks=None, tenant_id=TENANT,
        )

        assert _tak_terjawab(pesan), f"penanda tidak cocok dengan: {pesan!r}"

    async def test_pesan_saat_minggu_dipersempit_dikenali(self, pipa) -> None:
        pesan = await pipa._pesan_tak_terjawab(
            course_id="sbd", weeks=[1], tenant_id=TENANT,
        )

        assert _tak_terjawab(pesan), f"penanda tidak cocok dengan: {pesan!r}"

    async def test_pesan_saat_belum_ada_materi_dikenali(self, pipa) -> None:
        pipa._store.list_weeks = AsyncMock(return_value=[])

        pesan = await pipa._pesan_tak_terjawab(
            course_id="sbd", weeks=None, tenant_id=TENANT,
        )

        assert _tak_terjawab(pesan), f"penanda tidak cocok dengan: {pesan!r}"

    def test_pesan_jalur_guided_dikenali(self) -> None:
        """Jalur guided punya penolakannya sendiri — dua kalimat berbeda,
        tergantung mahasiswa sedang mempersempit minggu atau tidak."""
        dengan_minggu = RAGPipeline._pesan_tanpa_materi("normalisasi", [1, 2])
        tanpa_minggu = RAGPipeline._pesan_tanpa_materi("normalisasi", None)

        assert _tak_terjawab(dengan_minggu), f"tidak cocok: {dengan_minggu!r}"
        assert _tak_terjawab(tanpa_minggu), f"tidak cocok: {tanpa_minggu!r}"

    def test_jawaban_biasa_bukan_penolakan(self) -> None:
        assert not _tak_terjawab(
            "Normalisasi adalah proses menyusun tabel agar tidak ada redundansi."
        )

    def test_jawaban_navigasi_guided_bukan_kesenjangan_materi(self) -> None:
        """Alasan penanda dipakai, bukan `sources` yang kosong: jawaban navigasi
        juga tidak punya sumber, dan menghitungnya sebagai kesenjangan materi
        akan melaporkan angka yang salah ke dosen."""
        fakta = hitung_fakta([
            _rec(answer="Mata kuliah mana yang ingin kamu pelajari?", sumber=[]),
            _rec(answer="Pilih minggu keberapa?", sumber=[]),
        ])

        assert fakta.total == 2
        assert fakta.tak_terjawab == 0


class TestHitungan:
    def test_riwayat_kosong(self) -> None:
        fakta = hitung_fakta([])

        assert fakta.total == 0
        assert fakta.mahasiswa_unik == 0
        assert fakta.persen_tak_terjawab == 0.0
        assert fakta.sejak is None and fakta.hingga is None

    def test_mahasiswa_dihitung_dari_user_id_bukan_nama(self) -> None:
        """`student_name` teks bebas yang bisa diganti sendiri mahasiswa, jadi
        menghitungnya melaporkan satu orang yang berganti nama sebagai dua."""
        # Angkanya sengaja dibuat BERBEDA jauh antara dua cara menghitung: satu
        # orang (user_id 7) yang muncul dengan tiga nama. Versi pertama uji ini
        # memakai data yang kebetulan menghasilkan 2 pada kedua cara, sehingga
        # ia lulus walaupun kodenya menghitung dari nama.
        rekaman = [
            {**_rec(user_id=7), "student_name": "Budi"},
            {**_rec(user_id=7), "student_name": "Budi Santoso"},
            {**_rec(user_id=7), "student_name": "B."},
        ]

        assert hitung_fakta(rekaman).mahasiswa_unik == 1, (
            "tiga nama untuk satu user_id harus terhitung satu mahasiswa"
        )

    def test_rentang_waktu_dari_stempel_terkecil_dan_terbesar(self) -> None:
        fakta = hitung_fakta([
            _rec(timestamp="2026-10-20T10:00:00"),
            _rec(timestamp="2026-10-18T08:00:00"),
            _rec(timestamp="2026-10-22T15:00:00"),
        ])

        assert fakta.sejak == "2026-10-18T08:00:00"
        assert fakta.hingga == "2026-10-22T15:00:00"

    def test_berkas_dihitung_sekali_per_interaksi(self) -> None:
        """Yang diukur "berapa pertanyaan bersandar pada berkas ini", bukan
        berapa potongan terambil — tiga potongan dari satu berkas pada satu
        pertanyaan tetap satu."""
        fakta = hitung_fakta([
            _rec(sumber=["bab1.pdf", "bab1.pdf", "bab1.pdf", "bab2.pdf"]),
        ])

        assert dict(fakta.materi_teratas) == {"bab1.pdf": 1, "bab2.pdf": 1}

    def test_persen_tak_terjawab(self) -> None:
        fakta = hitung_fakta([
            _rec(answer="tidak mencakup informasi tentang X"),
            _rec(answer="Jawaban biasa."),
            _rec(answer="Jawaban biasa."),
            _rec(answer="Jawaban biasa."),
        ])

        assert fakta.tak_terjawab == 1
        assert fakta.persen_tak_terjawab == 25.0

    def test_topik_diurutkan_dari_yang_paling_sering(self) -> None:
        fakta = hitung_fakta([
            _rec(topik="normalisasi"), _rec(topik="normalisasi"),
            _rec(topik="normalisasi"), _rec(topik="indeks"),
        ])

        assert fakta.topik_teratas[0] == ("normalisasi", 3)


class TestPertanyaanBerulang:
    def test_kata_berbeda_isi_sama_dikelompokkan(self) -> None:
        """Bagi dosen, keduanya menunjukkan kebingungan yang sama."""
        a = _kunci_pertanyaan("apa itu primary key")
        b = _kunci_pertanyaan("primary key itu apa")

        assert a == b and a != ""

    def test_pertanyaan_berbeda_tidak_dikelompokkan(self) -> None:
        """Tanpa membuang kata umum, "apa itu X" dan "apa itu Y" berbagi tiga
        kata dari empat dan akan dihitung sebagai pertanyaan yang sama."""
        assert _kunci_pertanyaan("apa itu primary key") != _kunci_pertanyaan(
            "apa itu foreign key"
        )

    def test_pertanyaan_tanpa_kata_bermakna_tidak_dikelompokkan(self) -> None:
        """Kalau dikembalikan kunci kosong lalu dikelompokkan, seluruh sapaan
        pendek jadi satu kelompok dan dilaporkan "ditanyakan 40 kali"."""
        assert _kunci_pertanyaan("apa itu?") == ""
        assert _kunci_pertanyaan("...") == ""

        fakta = hitung_fakta([
            _rec(question="apa itu?", user_id=1),
            _rec(question="apa itu?", user_id=2),
        ])

        assert fakta.pertanyaan_berulang == []

    def test_hanya_yang_ditanyakan_beberapa_mahasiswa_berbeda(self) -> None:
        """Satu mahasiswa yang bertanya lima kali sedang kesulitan sendiri; dua
        mahasiswa yang bertanya hal sama menunjuk ke materinya — dan hanya yang
        kedua dapat ditindaklanjuti dosen di depan kelas."""
        fakta = hitung_fakta([
            _rec(question="apa itu normalisasi tabel", user_id=1),
            _rec(question="apa itu normalisasi tabel", user_id=1),
            _rec(question="apa itu normalisasi tabel", user_id=1),
            _rec(question="bagaimana cara membuat indeks basisdata", user_id=2),
            _rec(question="cara membuat indeks basisdata bagaimana", user_id=3),
        ])

        berulang = dict(fakta.pertanyaan_berulang)
        assert len(berulang) == 1
        assert next(iter(berulang.values())) == 2
        assert "indeks" in next(iter(berulang)).lower()

    def test_mahasiswa_tanpa_user_id_tidak_dihitung_sebagai_orang(self) -> None:
        """Log lama bisa tidak punya user_id. Menghitungnya sebagai orang yang
        berbeda akan melaporkan pertanyaan berulang yang tidak pernah terjadi."""
        fakta = hitung_fakta([
            _rec(question="apa itu normalisasi tabel", user_id=None),
            _rec(question="apa itu normalisasi tabel", user_id=None),
        ])

        assert fakta.pertanyaan_berulang == []


class TestBahanUntukModel:
    def test_angka_dan_contoh_ikut_di_bahan(self) -> None:
        bahan = susun_bahan(hitung_fakta([
            _rec(question="apa itu normalisasi tabel", topik="normalisasi",
                 tipe="definisi", sumber=["bab1.pdf"]),
            _rec(answer="tidak ada bagian materi yang menjawab pertanyaan itu",
                 question="siapa penemu sql", user_id=2),
        ]))

        assert "Jumlah interaksi: 2" in bahan
        assert "normalisasi" in bahan
        assert "bab1.pdf" in bahan
        assert "siapa penemu sql" in bahan, "contoh harus ikut supaya dosen bisa memeriksa"

    def test_bahan_tidak_tumbuh_sebanding_jumlah_chat(self) -> None:
        """Alasan model diberi angka, bukan log mentah: biayanya harus tetap
        walau riwayatnya sepuluh kali lebih panjang."""
        kecil = susun_bahan(hitung_fakta([
            _rec(question=f"pertanyaan nomor {i} tentang basisdata", user_id=i)
            for i in range(20)
        ]))
        besar = susun_bahan(hitung_fakta([
            _rec(question=f"pertanyaan nomor {i} tentang basisdata", user_id=i)
            for i in range(200)
        ]))

        assert len(besar) < len(kecil) * 2, (
            f"bahan tumbuh dari {len(kecil)} ke {len(besar)} karakter untuk "
            "10x interaksi — artinya log mentah ikut terkirim"
        )


class TestRingkasanLewatPipeline:
    """Tiga jaminan di tingkat pipeline, semuanya tentang BIAYA dan KEJUJURAN.

    Model hanya boleh dipanggil ketika ada yang bisa diringkas dan angkanya
    benar-benar berubah; dan kalau panggilannya gagal, angkanya tetap sampai ke
    dosen.
    """

    @pytest.fixture
    def pipa(self, monkeypatch):
        from src import pipeline as pipeline_mod

        p = RAGPipeline.__new__(RAGPipeline)
        gen = AsyncMock()
        gen.summarize_chat_history = AsyncMock(return_value="Narasi untuk dosen.")
        p._generator = gen

        rekaman: list[dict] = []

        def _baca(**kw):
            return list(rekaman), len(rekaman)

        monkeypatch.setattr(pipeline_mod, "read_conversation_logs", _baca)
        return p, gen, rekaman

    async def test_riwayat_kosong_tidak_memanggil_model(self, pipa) -> None:
        p, gen, _ = pipa

        hasil = await p.ringkasan_riwayat(content_id="kelas-1", tenant_id=TENANT)

        assert hasil["total_interaksi"] == 0
        assert hasil["ringkasan"] == ""
        gen.summarize_chat_history.assert_not_called()

    async def test_angka_tetap_terkirim_walau_narasinya_gagal(self, pipa) -> None:
        """Model gagal bukan alasan dosen kehilangan datanya — seluruh angka di
        balasan dihitung tanpa model sama sekali."""
        p, gen, rekaman = pipa
        rekaman.append(_rec(topik="normalisasi", sumber=["bab1.pdf"]))
        gen.summarize_chat_history = AsyncMock(return_value="")

        hasil = await p.ringkasan_riwayat(content_id="kelas-1", tenant_id=TENANT)

        assert hasil["ringkasan"] == ""
        assert hasil["total_interaksi"] == 1
        assert hasil["topik_teratas"] == [{"nama": "normalisasi", "jumlah": 1}]
        assert hasil["materi_teratas"] == [{"nama": "bab1.pdf", "jumlah": 1}]

    async def test_angka_yang_sama_tidak_memanggil_model_dua_kali(self, pipa) -> None:
        p, gen, rekaman = pipa
        rekaman.append(_rec(topik="normalisasi"))

        pertama = await p.ringkasan_riwayat(content_id="kelas-1", tenant_id=TENANT)
        kedua = await p.ringkasan_riwayat(content_id="kelas-1", tenant_id=TENANT)

        assert pertama["dari_cache"] is False
        assert kedua["dari_cache"] is True
        assert kedua["ringkasan"] == "Narasi untuk dosen."
        assert gen.summarize_chat_history.await_count == 1

    async def test_chat_baru_menerbitkan_ulang_ringkasannya(self, pipa) -> None:
        """Cache dikunci pada sidik jari BAHANNYA, bukan waktu kedaluwarsa, jadi
        dosen tidak perlu menebak kapan datanya segar."""
        p, gen, rekaman = pipa
        rekaman.append(_rec(topik="normalisasi"))

        await p.ringkasan_riwayat(content_id="kelas-1", tenant_id=TENANT)
        rekaman.append(_rec(topik="indeks", user_id=2))
        hasil = await p.ringkasan_riwayat(content_id="kelas-1", tenant_id=TENANT)

        assert hasil["dari_cache"] is False
        assert gen.summarize_chat_history.await_count == 2

    async def test_kampus_lain_tidak_memakai_ringkasan_yang_sama(self, pipa) -> None:
        """Ringkasan satu kampus tidak boleh tersaji ke kampus lain — kebocoran
        yang tidak akan terlihat sebagai galat, hanya sebagai angka yang aneh.

        Diukur dengan mematikan penjagaannya: uji ini TETAP lulus kalau hanya
        `tenant_id` yang dikeluarkan dari kunci cache, karena `gen_cache.load()`
        masih memeriksa ulang pemilik entrinya. Ia baru merah ketika KEDUA lapis
        dimatikan. Jadi yang dijaga uji ini perilakunya, dan perlindungannya
        memang berlapis dua — bukan satu titik yang kalau terlewat langsung
        bocor.
        """
        p, gen, rekaman = pipa
        rekaman.append(_rec(topik="normalisasi"))

        await p.ringkasan_riwayat(content_id="kelas-1", tenant_id="kampus-a")
        hasil = await p.ringkasan_riwayat(content_id="kelas-1", tenant_id="kampus-b")

        assert hasil["dari_cache"] is False
        assert gen.summarize_chat_history.await_count == 2
