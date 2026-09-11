"""Uji isolasi antar tenant.

Berkas ini adalah bukti pendukung satu klaim yang dijual kepada pelanggan:
**materi dan riwayat satu kampus tidak dapat terlihat oleh kampus lain.**

Karena itu polanya selalu sama — dua tenant palsu, `kampus-a` dan `kampus-b`,
diberi data yang mirip (nama mata kuliah sama, `content_id` sama, `student_id`
sama), lalu diperiksa bahwa tidak satu pun jalur baca mempertemukan keduanya.
Kesamaan itu disengaja: kalau isolasi bocor, tabrakan nama adalah bentuk
kebocoran yang paling mungkin terjadi di dunia nyata, sebab pola penamaan
`<matkul>-minggu-<n>` memang dianjurkan kepada semua pelanggan.

Uji pada `TestQdrantIsolation` memakai Qdrant sungguhan dalam mode memori, bukan
tiruan. Menirukan penyimpanan berarti menirukan pula penyaringnya — dan
penyaring itulah yang sedang diuji, jadi hasilnya tidak akan membuktikan apa pun.
"""
from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from src import gen_cache
from src.api.session import Session, SessionStore
from src.schemas import Chunk, ElementType
from src.security import audit, crypto, keys, redact
from src.security.ratelimit import AuthThrottle, RateLimiter
from src.storage import content_store, conversation_store, tenant_store
from src.storage.qdrant_store import TENANT_FIELD, QdrantStore
from src.tenancy import (
    ALL_SCOPES,
    READER_SCOPES,
    SCOPE_CHAT,
    SCOPE_CONTENT_WRITE,
    TenantContext,
    TenantPermissionError,
    TenantScopeError,
    cache_key,
    is_valid_tenant_id,
    require_tenant_id,
)

A = "kampus-a"
B = "kampus-b"


# --- fondasi ---------------------------------------------------------------

class TestTenantIdValidation:
    """tenant_id ikut menjadi nama direktori, jadi bentuknya harus ketat."""

    @pytest.mark.parametrize("nilai", ["kampus-a", "ui", "itb_2024", "a1"])
    def test_bentuk_sah_diterima(self, nilai: str) -> None:
        assert is_valid_tenant_id(nilai)

    @pytest.mark.parametrize(
        "nilai",
        [
            "",            # kosong
            None,          # tidak dikirim
            "a",           # terlalu pendek
            "Kampus-A",    # huruf besar → dua nama untuk satu direktori di Windows
            "kampus a",    # spasi
            "../etc",      # keluar direktori
            "kampus/a",    # pemisah path
            "-kampus",     # diawali strip
            "k" * 64,      # melebihi batas
        ],
    )
    def test_bentuk_berbahaya_ditolak(self, nilai: str | None) -> None:
        assert not is_valid_tenant_id(nilai)
        with pytest.raises(TenantScopeError):
            require_tenant_id(nilai)

    def test_pesan_galat_menyebut_operasinya(self) -> None:
        """Galat harus menuntun langsung ke jalur kode yang lupa menyaring."""
        with pytest.raises(TenantScopeError, match="search"):
            require_tenant_id(None, operation="search")


class TestTenantContext:
    def test_konteks_menolak_tenant_tak_sah(self) -> None:
        with pytest.raises(TenantScopeError):
            TenantContext(tenant_id="../lain")

    def test_hak_akses_ditegakkan(self) -> None:
        pembaca = TenantContext(tenant_id=A, scopes=READER_SCOPES)
        assert pembaca.has(SCOPE_CHAT)
        assert not pembaca.has(SCOPE_CONTENT_WRITE)
        with pytest.raises(TenantPermissionError):
            pembaca.require(SCOPE_CONTENT_WRITE)

    def test_abac_membatasi_mata_kuliah(self) -> None:
        t = TenantContext(tenant_id=A, allowed_courses=frozenset({"sbd"}))
        assert t.may_access_course("sbd")
        assert not t.may_access_course("kka")
        with pytest.raises(TenantPermissionError):
            t.require_course("kka")

    def test_abac_menolak_kueri_tanpa_mata_kuliah(self) -> None:
        """Tanpa mata kuliah, kueri mencakup SELURUH milik tenant.

        Itu lebih luas daripada jatah kunci yang dibatasi, jadi harus ditolak —
        bukan diloloskan karena "tidak menyebut mata kuliah terlarang".
        """
        t = TenantContext(tenant_id=A, allowed_courses=frozenset({"sbd"}))
        assert not t.may_access_course(None)

    def test_tanpa_batasan_boleh_semua(self) -> None:
        assert TenantContext(tenant_id=A).may_access_course("apa saja")

    def test_bentuk_untuk_log_tidak_memuat_kunci(self) -> None:
        t = TenantContext(tenant_id=A, key_id="abc123")
        assert t.redacted() == {"tenant_id": A, "key_id": "abc123"}

    def test_cache_key_selalu_diawali_tenant(self) -> None:
        assert cache_key(A, "quiz", "bab1.pdf") != cache_key(B, "quiz", "bab1.pdf")


# --- kunci API -------------------------------------------------------------

@pytest.fixture(autouse=True)
def _pepper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pepper tetap agar hash dapat diulang di dalam satu uji."""
    from pydantic import SecretStr

    from src.config import settings
    monkeypatch.setattr(
        settings, "tenant_key_pepper", SecretStr("pepper-uji"), raising=False,
    )


class TestApiKeys:
    def test_terbit_lalu_terverifikasi(self) -> None:
        k = keys.generate_key()
        terurai = keys.parse_key(k.raw)
        assert terurai is not None
        key_id, secret = terurai
        assert key_id == k.key_id
        assert keys.verify_secret(secret, k.secret_hash)

    def test_rahasia_tidak_tersimpan_apa_adanya(self) -> None:
        """Yang disimpan hanya hash — bocornya berkas tenant tidak memberi kunci."""
        k = keys.generate_key()
        _, secret = keys.parse_key(k.raw)  # type: ignore[misc]
        assert secret not in k.secret_hash
        assert k.secret_hash != secret

    def test_dua_kunci_tidak_pernah_sama(self) -> None:
        assert len({keys.generate_key().raw for _ in range(50)}) == 50

    def test_rahasia_salah_ditolak(self) -> None:
        k = keys.generate_key()
        assert not keys.verify_secret("bukan-rahasianya", k.secret_hash)

    def test_pepper_berbeda_membatalkan_kunci(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Hash tanpa pepper tidak berlaku di server lain — itu memang gunanya."""
        from pydantic import SecretStr

        from src.config import settings
        k = keys.generate_key()
        _, secret = keys.parse_key(k.raw)  # type: ignore[misc]
        monkeypatch.setattr(settings, "tenant_key_pepper", SecretStr("pepper-lain"))
        assert not keys.verify_secret(secret, k.secret_hash)

    @pytest.mark.parametrize(
        "mentah",
        [
            None, "", "bukan-kunci", "ragk_pendek.x", "ragk_.rahasia",
            f"ragk_{'z' * 16}.rahasia",             # key_id bukan hex
            "Bearer ragk_0123456789abcdef.rahasia",  # awalan salah
        ],
    )
    def test_bentuk_kunci_salah_ditolak_sebelum_menyentuh_penyimpanan(
        self, mentah: str | None,
    ) -> None:
        assert keys.parse_key(mentah) is None

    def test_hash_kata_sandi_tidak_dapat_dibalik(self) -> None:
        h = keys.hash_password("rahasia-dosen")
        assert "rahasia-dosen" not in h
        assert keys.verify_password("rahasia-dosen", h)
        assert not keys.verify_password("salah", h)


class TestTenantStore:
    @pytest.fixture(autouse=True)
    def _dir(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tenant_store, "TENANT_DIR", tmp_path / "tenants")
        tenant_store.invalidate_cache()

    def test_autentikasi_menghasilkan_tenant_yang_benar(self) -> None:
        tenant_store.create_tenant(A, name="Kampus A")
        tenant_store.create_tenant(B, name="Kampus B")
        kunci_a = tenant_store.issue_key(A, label="BE A")

        ctx = tenant_store.authenticate(kunci_a.raw)
        assert ctx is not None
        assert ctx.tenant_id == A
        assert ctx.key_id == kunci_a.key_id

    def test_kunci_tenant_a_tidak_pernah_menghasilkan_tenant_b(self) -> None:
        """Inti seluruh isolasi: identitas hanya lahir dari kunci."""
        tenant_store.create_tenant(A)
        tenant_store.create_tenant(B)
        kunci_a = tenant_store.issue_key(A)
        kunci_b = tenant_store.issue_key(B)

        assert tenant_store.authenticate(kunci_a.raw).tenant_id == A   # type: ignore[union-attr]
        assert tenant_store.authenticate(kunci_b.raw).tenant_id == B   # type: ignore[union-attr]

    def test_kunci_dicabut_langsung_tidak_berlaku(self) -> None:
        tenant_store.create_tenant(A)
        k = tenant_store.issue_key(A)
        assert tenant_store.authenticate(k.raw) is not None

        assert tenant_store.revoke_key(A, k.key_id)
        assert tenant_store.authenticate(k.raw) is None

    def test_tenant_dibekukan_kehilangan_seluruh_aksesnya(self) -> None:
        """Satu tuas untuk memutus pelanggan yang menunggak, tanpa mencabut satu per satu."""
        tenant_store.create_tenant(A)
        k = tenant_store.issue_key(A)
        tenant_store.set_status(A, tenant_store.STATUS_SUSPENDED)
        assert tenant_store.authenticate(k.raw) is None

    def test_kunci_asing_ditolak(self) -> None:
        tenant_store.create_tenant(A)
        tenant_store.issue_key(A)
        assert tenant_store.authenticate("ragk_0123456789abcdef.palsu") is None

    def test_hak_melekat_pada_kunci_bukan_tenant(self) -> None:
        """Satu tenant boleh punya kunci unggah dan kunci baca-saja sekaligus."""
        tenant_store.create_tenant(A)
        penulis = tenant_store.issue_key(A, scopes=set(ALL_SCOPES))
        pembaca = tenant_store.issue_key(A, scopes=set(READER_SCOPES))

        assert tenant_store.authenticate(penulis.raw).has(SCOPE_CONTENT_WRITE)   # type: ignore[union-attr]
        assert not tenant_store.authenticate(pembaca.raw).has(SCOPE_CONTENT_WRITE)  # type: ignore[union-attr]

    def test_rahasia_tidak_pernah_ditulis_ke_disk(self) -> None:
        tenant_store.create_tenant(A)
        k = tenant_store.issue_key(A)
        _, secret = keys.parse_key(k.raw)  # type: ignore[misc]
        isi = (tenant_store.TENANT_DIR / f"{A}.json").read_text(encoding="utf-8")
        assert secret not in isi
        assert k.raw not in isi

    def test_berkas_rusak_tidak_mematikan_autentikasi_tenant_lain(self) -> None:
        """Satu berkas gagal baca tidak boleh menjatuhkan seluruh layanan."""
        tenant_store.create_tenant(A)
        k = tenant_store.issue_key(A)
        (tenant_store.TENANT_DIR / "rusak.json").write_text("{bukan json")
        tenant_store.invalidate_cache()
        assert tenant_store.authenticate(k.raw) is not None


# --- Qdrant: inti isolasi data ---------------------------------------------

def _chunk(chunk_id: str, *, course: str, week: int, source: str, teks: str) -> Chunk:
    """Potongan dengan vektor tiruan — embedder sungguhan tidak diperlukan di sini.

    Qdrant menuntut id titik berupa UUID, jadi label yang mudah dibaca ("a1")
    dipetakan ke UUID secara deterministik agar tetap dapat dirujuk di uji.
    """
    return Chunk(
        chunk_id=str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id)),
        text=teks,
        parent_element_id="e1",
        element_type=ElementType.TEXT,
        source_file=source,
        page_number=1,
        content_id=f"{course}-minggu-{week}",
        course_id=course,
        course_name=course.upper(),
        week=week,
        dense_embedding=[0.1] * 1024,
        sparse_embedding=None,
    )


class TestQdrantIsolation:
    """Qdrant sungguhan (mode memori) dengan data dua tenant di satu koleksi."""

    @pytest.fixture
    async def store(self) -> QdrantStore:
        s = QdrantStore(
            client=QdrantClient(":memory:"),
            collection="uji_tenancy",
            enable_sparse=False,
        )
        await s.ensure_collection()
        # Sengaja identik: nama mata kuliah, minggu, dan nama berkas yang sama
        # di dua tenant. Inilah keadaan yang membuat kebocoran paling mungkin.
        await s.upsert_chunks(
            [_chunk("a1", course="sbd", week=1, source="bab1.pdf",
                    teks="Normalisasi basis data kampus A")],
            tenant_id=A,
        )
        await s.upsert_chunks(
            [_chunk("b1", course="sbd", week=1, source="bab1.pdf",
                    teks="Normalisasi basis data kampus B")],
            tenant_id=B,
        )
        return s

    async def test_pencarian_tidak_melintasi_tenant(self, store: QdrantStore) -> None:
        """Tanpa satu pun penyaring lain — keadaan tanya-bebas mahasiswa."""
        hasil = await store.search([0.1] * 1024, top_k=50, tenant_id=A)
        assert hasil, "tenant A harus tetap menemukan materinya sendiri"
        assert all(r["payload"][TENANT_FIELD] == A for r in hasil)
        assert not any("kampus B" in r["payload"]["text"] for r in hasil)

    async def test_katalog_hanya_memuat_mata_kuliah_sendiri(
        self, store: QdrantStore,
    ) -> None:
        await store.upsert_chunks(
            [_chunk("b2", course="kka", week=1, source="kka1.pdf", teks="Kalkulus")],
            tenant_id=B,
        )
        assert [c["course_id"] for c in await store.list_courses(tenant_id=A)] == ["sbd"]
        assert [c["course_id"] for c in await store.list_courses(tenant_id=B)] == [
            "kka", "sbd",
        ]

    async def test_minggu_dan_materi_terpisah(self, store: QdrantStore) -> None:
        await store.upsert_chunks(
            [_chunk("b3", course="sbd", week=9, source="bab9.pdf", teks="Indeks")],
            tenant_id=B,
        )
        assert await store.list_weeks("sbd", tenant_id=A) == [1]
        assert await store.list_weeks("sbd", tenant_id=B) == [1, 9]

    async def test_teks_materi_nama_sama_tidak_tertukar(
        self, store: QdrantStore,
    ) -> None:
        """Dua tenant, `content_id` dan nama berkas identik — isinya harus beda."""
        teks_a = await store.get_material_text("sbd-minggu-1", "bab1.pdf", tenant_id=A)
        teks_b = await store.get_material_text("sbd-minggu-1", "bab1.pdf", tenant_id=B)
        assert "kampus A" in teks_a and "kampus B" not in teks_a
        assert "kampus B" in teks_b and "kampus A" not in teks_b

    async def test_penghapusan_tidak_menyentuh_tenant_lain(
        self, store: QdrantStore,
    ) -> None:
        """Nama berkas bertabrakan adalah hal biasa; menghapus milik orang lain tidak."""
        await store.delete_by_source("bab1.pdf", tenant_id=A)
        assert await store.search([0.1] * 1024, top_k=50, tenant_id=A) == []
        assert await store.search([0.1] * 1024, top_k=50, tenant_id=B), \
            "materi tenant B ikut terhapus — kehilangan data lintas pelanggan"

    async def test_hapus_seluruh_data_tenant(self, store: QdrantStore) -> None:
        await store.delete_tenant_data(tenant_id=A)
        assert await store.list_courses(tenant_id=A) == []
        assert await store.list_courses(tenant_id=B) != []

    async def test_menulis_potongan_bertenant_lain_ditolak(
        self, store: QdrantStore,
    ) -> None:
        """Percampuran di lapisan atas harus berhenti sebelum tertulis ke index."""
        nyasar = _chunk("x1", course="sbd", week=1, source="x.pdf", teks="x")
        nyasar = nyasar.model_copy(update={"tenant_id": B})
        with pytest.raises(TenantScopeError):
            await store.upsert_chunks([nyasar], tenant_id=A)

    async def test_setiap_potongan_tercap_tenant(self, store: QdrantStore) -> None:
        hasil = await store.search([0.1] * 1024, top_k=50, tenant_id=B)
        assert all(r["payload"].get(TENANT_FIELD) == B for r in hasil)


class TestQdrantFilterSelaluBertenant:
    """Penjagaan struktural: tidak ada jalan membangun filter tanpa tenant."""

    def test_filter_kosong_tetap_menyaring_tenant(self) -> None:
        """Dulu keadaan ini menghasilkan None — Qdrant membacanya 'cari semua'."""
        f = QdrantStore._build_filter(tenant_id=A)
        assert f is not None
        assert len(f.must) == 1
        assert f.must[0].key == TENANT_FIELD
        assert f.must[0].match.value == A

    def test_tenant_ikut_bersama_penyaring_lain(self) -> None:
        f = QdrantStore._build_filter(
            tenant_id=A, course_id="sbd", weeks=[1, 2], source_filter="bab1.pdf",
        )
        assert TENANT_FIELD in {c.key for c in f.must}

    def test_tanpa_tenant_gagal_bukan_mencari_semua(self) -> None:
        with pytest.raises(TenantScopeError):
            QdrantStore._build_filter(tenant_id="")

    @pytest.mark.parametrize(
        "metode, argumen",
        [
            ("search", ([0.1] * 1024,)),
            ("list_courses", ()),
            ("list_weeks", ("sbd",)),
            ("list_materials", ("sbd", 1)),
            ("list_materials_for_weeks", ("sbd", [1])),
            ("get_week_text", ("sbd", [1])),
            ("get_material_text", ("c1", "bab1.pdf")),
            ("list_indexed_files", ()),
            ("delete_by_source", ("bab1.pdf",)),
            ("delete_stale_chunks", ("bab1.pdf", "c1", ["id-baru"])),
            ("delete_tenant_data", ()),
        ],
    )
    async def test_semua_metode_menolak_tenant_kosong(
        self, metode: str, argumen: tuple,
    ) -> None:
        """Sapuan menyeluruh: metode baru yang lupa menyaring akan tertangkap di sini."""
        s = QdrantStore(client=QdrantClient(":memory:"), collection="x",
                        enable_sparse=False)
        with pytest.raises(TenantScopeError):
            await getattr(s, metode)(*argumen, tenant_id="")


# --- percakapan ------------------------------------------------------------

class TestConversationIsolation:
    @pytest.fixture(autouse=True)
    def _dir(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(conversation_store, "CONVERSATION_DIR", tmp_path / "conv")

    def test_tenant_lain_tidak_dapat_membaca(self) -> None:
        conversation_store.save(
            {"conversation_id": "rahasia", "title": "Rencana ujian"}, tenant_id=A,
        )
        assert conversation_store.load("rahasia", tenant_id=A) is not None
        assert conversation_store.load("rahasia", tenant_id=B) is None

    def test_tenant_lain_tidak_dapat_menghapus(self) -> None:
        conversation_store.save({"conversation_id": "x", "title": "T"}, tenant_id=A)
        assert conversation_store.delete("x", tenant_id=B) is False
        assert conversation_store.load("x", tenant_id=A) is not None

    def test_daftar_terpisah_per_tenant(self) -> None:
        conversation_store.save({"conversation_id": "a1", "title": "A"}, tenant_id=A)
        conversation_store.save({"conversation_id": "b1", "title": "B"}, tenant_id=B)
        assert [r["title"] for r in conversation_store.list_summaries(tenant_id=A)] == ["A"]
        assert [r["title"] for r in conversation_store.list_summaries(tenant_id=B)] == ["B"]

    def test_id_sama_di_dua_tenant_tidak_saling_menimpa(self) -> None:
        """`session_id` dibuat server, tetapi kesamaan tidak boleh berakibat fatal."""
        conversation_store.save({"conversation_id": "sama", "title": "Milik A"}, tenant_id=A)
        conversation_store.save({"conversation_id": "sama", "title": "Milik B"}, tenant_id=B)
        assert conversation_store.load("sama", tenant_id=A)["title"] == "Milik A"
        assert conversation_store.load("sama", tenant_id=B)["title"] == "Milik B"

    def test_student_id_sama_tidak_tercampur(self) -> None:
        """Satu NIM yang kebetulan sama di dua kampus tetap dua orang berbeda."""
        conversation_store.save(
            {"conversation_id": "a1", "student_id": "2021001", "title": "A"}, tenant_id=A,
        )
        conversation_store.save(
            {"conversation_id": "b1", "student_id": "2021001", "title": "B"}, tenant_id=B,
        )
        hasil = conversation_store.list_summaries(student_id="2021001", tenant_id=A)
        assert [r["title"] for r in hasil] == ["A"]

    def test_student_id_tidak_tersimpan_polos_sebagai_penyaring(self) -> None:
        """Indeks buta: penyaringan berjalan tanpa menyimpan nilai yang bisa dibalik."""
        conversation_store.save(
            {"conversation_id": "a1", "student_id": "2021001"}, tenant_id=A,
        )
        isi = (conversation_store.tenant_dir(A) / "a1.json").read_text(encoding="utf-8")
        assert "student_ref" in isi
        # Indeks buta kedua tenant berbeda walau nilainya sama.
        assert crypto.blind_index("2021001", tenant_id=A) != \
               crypto.blind_index("2021001", tenant_id=B)

    def test_berkas_salah_tempat_ditolak(self) -> None:
        """Pemulihan cadangan yang keliru tidak boleh menyajikan data tenant lain."""
        conversation_store.save({"conversation_id": "x", "title": "T"}, tenant_id=A)
        salah = conversation_store.tenant_dir(A) / "x.json"
        tujuan = conversation_store.tenant_dir(B)
        tujuan.mkdir(parents=True, exist_ok=True)
        (tujuan / "x.json").write_text(salah.read_text(encoding="utf-8"), encoding="utf-8")
        assert conversation_store.load("x", tenant_id=B) is None

    def test_id_berbahaya_ditolak(self) -> None:
        for jahat in ["../../etc/passwd", "a/b", "..", ""]:
            assert conversation_store.load(jahat, tenant_id=A) is None

    def test_hapus_seluruh_percakapan_satu_tenant(self) -> None:
        conversation_store.save({"conversation_id": "a1"}, tenant_id=A)
        conversation_store.save({"conversation_id": "b1"}, tenant_id=B)
        assert conversation_store.delete_all_for_tenant(tenant_id=A) == 1
        assert conversation_store.list_summaries(tenant_id=B)


# --- berkas ----------------------------------------------------------------

class TestContentStoreIsolation:
    @pytest.fixture(autouse=True)
    def _root(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            type(content_store.settings), "storage_path",
            property(lambda _: tmp_path / "storage"),
        )

    def test_direktori_terpisah_secara_fisik(self) -> None:
        ra = content_store.storage_root(tenant_id=A)
        rb = content_store.storage_root(tenant_id=B)
        assert ra != rb
        assert A in str(ra) and B in str(rb)

    def test_daftar_isi_tidak_bocor(self) -> None:
        (content_store.storage_root(tenant_id=A) / "sbd-minggu-1").mkdir(parents=True)
        (content_store.storage_root(tenant_id=B) / "kka-minggu-1").mkdir(parents=True)
        assert content_store.list_contents(tenant_id=A) == ["sbd-minggu-1"]
        assert content_store.list_contents(tenant_id=B) == ["kka-minggu-1"]

    def test_content_id_sama_tidak_saling_menimpa(self) -> None:
        """Pola `<matkul>-minggu-<n>` dianjurkan, jadi tabrakan ini pasti terjadi."""
        for tenant, isi in ((A, "materi A"), (B, "materi B")):
            d = content_store.storage_root(tenant_id=tenant) / "sbd-minggu-1"
            d.mkdir(parents=True)
            (d / "bab1.pdf").write_text(isi, encoding="utf-8")

        pa = content_store.resolve_file("sbd-minggu-1", "bab1.pdf", tenant_id=A)
        pb = content_store.resolve_file("sbd-minggu-1", "bab1.pdf", tenant_id=B)
        assert pa.read_text(encoding="utf-8") == "materi A"
        assert pb.read_text(encoding="utf-8") == "materi B"

    @pytest.mark.parametrize(
        "jahat", ["../kampus-b", "..", "a/b", "", ".tersembunyi", "a" * 200],
    )
    def test_content_id_berbahaya_ditolak(self, jahat: str) -> None:
        assert content_store.content_dir(jahat, tenant_id=A) is None

    @pytest.mark.parametrize(
        "jahat",
        ["../../../etc/passwd", "..\\..\\windows\\win.ini", "../bab1.pdf", ""],
    )
    def test_nama_berkas_tidak_dapat_keluar_direktori(self, jahat: str) -> None:
        d = content_store.storage_root(tenant_id=A) / "c1"
        d.mkdir(parents=True)
        (d / "bab1.pdf").write_text("isi", encoding="utf-8")
        assert content_store.resolve_file("c1", jahat, tenant_id=A) is None

    def test_pemakaian_dihitung_per_tenant(self) -> None:
        d = content_store.storage_root(tenant_id=A) / "c1"
        d.mkdir(parents=True)
        (d / "f.bin").write_bytes(b"x" * 100)
        assert content_store.tenant_usage_bytes(tenant_id=A) == 100
        assert content_store.tenant_usage_bytes(tenant_id=B) == 0


# --- cache hasil LLM -------------------------------------------------------

class TestGenCacheIsolation:
    @pytest.fixture(autouse=True)
    def _dir(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gen_cache, "_CACHE_DIR", tmp_path / "cache")

    def test_kuis_tidak_dipakai_bersama(self) -> None:
        """Kebocoran di sini tak akan tertahan penyaring pencarian.

        Setelah tersimpan, isi cache tidak pernah melewati Qdrant lagi — jadi
        kunci cache-lah satu-satunya yang memisahkan kedua tenant.
        """
        H = gen_cache.hash_material("isi bab 1")
        gen_cache.save(
            "quiz", "sbd-minggu-1", "bab1.pdf", ["soal A"], tenant_id=A, material_hash=H,
        )
        assert gen_cache.load(
            "quiz", "sbd-minggu-1", "bab1.pdf", tenant_id=A, material_hash=H,
        ) == ["soal A"]
        assert gen_cache.load(
            "quiz", "sbd-minggu-1", "bab1.pdf", tenant_id=B, material_hash=H,
        ) is None

    def test_kedua_tenant_menyimpan_versinya_sendiri(self) -> None:
        H = gen_cache.hash_material("isi f.pdf")
        gen_cache.save("starter", "c1", "f.pdf", ["A"], tenant_id=A, material_hash=H)
        gen_cache.save("starter", "c1", "f.pdf", ["B"], tenant_id=B, material_hash=H)
        assert gen_cache.load(
            "starter", "c1", "f.pdf", tenant_id=A, material_hash=H,
        ) == ["A"]
        assert gen_cache.load(
            "starter", "c1", "f.pdf", tenant_id=B, material_hash=H,
        ) == ["B"]

    def test_materi_berubah_membatalkan_entri_lama(self) -> None:
        """Sidik jari materi ikut jadi kunci, jadi entri lama berhenti terpakai.

        Ini yang menjaga agar perbaikan pada perakit teks materi benar-benar
        terasa: tanpa sidik jari, soal yang disusun dari materi versi lama akan
        terus tersaji meski materinya sudah diperbaiki.
        """
        lama = gen_cache.hash_material("materi versi lama")
        baru = gen_cache.hash_material("materi versi baru")
        gen_cache.save(
            "quiz", "c1", "f.pdf", ["soal lama"], tenant_id=A, material_hash=lama,
        )
        assert gen_cache.load(
            "quiz", "c1", "f.pdf", tenant_id=A, material_hash=lama,
        ) == ["soal lama"]
        assert gen_cache.load(
            "quiz", "c1", "f.pdf", tenant_id=A, material_hash=baru,
        ) is None

    def test_tanpa_tenant_gagal(self) -> None:
        with pytest.raises(TenantScopeError):
            gen_cache.load(
                "quiz", "c1", "f.pdf", tenant_id="", material_hash=gen_cache.hash_material("x"),
            )


# --- session ---------------------------------------------------------------

class TestSessionIsolation:
    @pytest.fixture(autouse=True)
    def _dir(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(conversation_store, "CONVERSATION_DIR", tmp_path / "conv")

    def test_session_tenant_lain_tidak_terbaca(self) -> None:
        store = SessionStore()
        s = store.create(tenant_id=A, student_id="2021001")
        s.record("user", "apa itu normalisasi")
        s.persist()

        assert store.get(s.session_id, tenant_id=A) is not None
        assert store.get(s.session_id, tenant_id=B) is None

    def test_session_id_bocor_tidak_dapat_dipakai_tenant_lain(self) -> None:
        """`session_id` bolak-balik lewat klien; ia harus dianggap bisa dicuri."""
        store = SessionStore()
        s = store.create(tenant_id=A)
        s.persist()

        pulih = store.get_or_create(s.session_id, tenant_id=B)
        assert pulih.session_id != s.session_id, \
            "tenant B mengambil alih session tenant A"
        assert pulih.tenant_id == B

    def test_cache_memori_terpisah(self) -> None:
        """Cache berkunci tunggal akan melewati pemisahan direktori di disk."""
        store = SessionStore()
        a = store.create(tenant_id=A)
        assert store.get(a.session_id, tenant_id=B) is None

    def test_session_menolak_tenant_kosong(self) -> None:
        with pytest.raises(TenantScopeError):
            SessionStore().create(tenant_id="")

    def test_tenant_diambil_dari_kredensial_bukan_berkas(self) -> None:
        """Berkas menentukan isinya, bukan siapa yang berhak membacanya."""
        s = Session.from_record({"conversation_id": "x", "tenant_id": B}, tenant_id=A)
        assert s.tenant_id == A


# --- jejak audit -----------------------------------------------------------

class TestAuditChain:
    @pytest.fixture(autouse=True)
    def _dir(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(audit, "AUDIT_DIR", tmp_path / "audit")
        audit._tail_cache.clear()

    def test_rantai_utuh_setelah_beberapa_peristiwa(self) -> None:
        for i in range(5):
            audit.record(audit.CHAT_ASK, tenant_id=A, detail={"n": i})
        berkas = next((audit.AUDIT_DIR / A).glob("*.jsonl"))
        utuh, jumlah, _ = audit.verify_chain(berkas)
        assert utuh and jumlah == 5

    def test_perubahan_isi_terdeteksi(self) -> None:
        audit.record(audit.CONTENT_UPLOAD, tenant_id=A, detail={"file": "asli.pdf"})
        audit.record(audit.CONTENT_UPLOAD, tenant_id=A, detail={"file": "kedua.pdf"})
        berkas = next((audit.AUDIT_DIR / A).glob("*.jsonl"))

        baris = berkas.read_text(encoding="utf-8").splitlines()
        baris[0] = baris[0].replace("asli.pdf", "palsu.pdf")
        berkas.write_text("\n".join(baris) + "\n", encoding="utf-8")

        utuh, _, ket = audit.verify_chain(berkas)
        assert not utuh and "MAC" in ket

    def test_penghapusan_baris_terdeteksi(self) -> None:
        """Menghapus jejak sama pentingnya untuk terdeteksi seperti mengubahnya."""
        for i in range(3):
            audit.record(audit.AUTH_SUCCESS, tenant_id=A, detail={"n": i})
        berkas = next((audit.AUDIT_DIR / A).glob("*.jsonl"))
        baris = berkas.read_text(encoding="utf-8").splitlines()
        del baris[1]
        berkas.write_text("\n".join(baris) + "\n", encoding="utf-8")

        utuh, _, ket = audit.verify_chain(berkas)
        assert not utuh and "rantai" in ket

    def test_jejak_terpisah_per_tenant(self) -> None:
        audit.record(audit.CHAT_ASK, tenant_id=A)
        audit.record(audit.CHAT_ASK, tenant_id=B)
        assert (audit.AUDIT_DIR / A).exists()
        assert (audit.AUDIT_DIR / B).exists()

    def test_kunci_tidak_pernah_masuk_jejak(self) -> None:
        audit.record(
            audit.AUTH_FAILURE, tenant_id=A,
            detail={"api_key": "ragk_0123456789abcdef.sangat-rahasia"},
        )
        isi = next((audit.AUDIT_DIR / A).glob("*.jsonl")).read_text(encoding="utf-8")
        assert "sangat-rahasia" not in isi

    def test_peristiwa_tanpa_tenant_tetap_tercatat(self) -> None:
        """Autentikasi gagal belum punya tenant, tetapi justru paling perlu dicatat."""
        audit.record(audit.AUTH_FAILURE, ip="1.2.3.4")
        assert (audit.AUDIT_DIR / audit.SYSTEM_SCOPE).exists()


# --- pembatasan laju -------------------------------------------------------

class TestRateLimit:
    def test_batas_per_tenant_ditegakkan(self) -> None:
        lim = RateLimiter()
        for _ in range(5):
            assert lim.check(
                ip=None, key_id=None, tenant_id=A,
                per_minute=5, per_day=100, ip_per_minute=0,
            ).allowed
        assert not lim.check(
            ip=None, key_id=None, tenant_id=A,
            per_minute=5, per_day=100, ip_per_minute=0,
        ).allowed

    def test_tenant_yang_boros_tidak_menghabiskan_jatah_tenant_lain(self) -> None:
        """Kalau jatah dipakai bersama, satu pelanggan dapat menjatuhkan yang lain."""
        lim = RateLimiter()
        for _ in range(5):
            lim.check(ip=None, key_id=None, tenant_id=A,
                      per_minute=5, per_day=100, ip_per_minute=0)
        assert lim.check(
            ip=None, key_id=None, tenant_id=B,
            per_minute=5, per_day=100, ip_per_minute=0,
        ).allowed

    def test_penolakan_menyebut_lama_menunggu(self) -> None:
        lim = RateLimiter()
        for _ in range(2):
            lim.check(ip="1.1.1.1", key_id=None, tenant_id=None,
                      per_minute=0, per_day=0, ip_per_minute=2)
        putusan = lim.check(ip="1.1.1.1", key_id=None, tenant_id=None,
                            per_minute=0, per_day=0, ip_per_minute=2)
        assert not putusan.allowed and putusan.retry_after >= 1

    def test_perlambatan_naik_setelah_gagal_berulang(self) -> None:
        t = AuthThrottle(threshold=3, max_delay=60.0)
        for _ in range(3):
            t.record_failure("1.2.3.4")
        _, tunggu1 = t.penalty("1.2.3.4")
        t.record_failure("1.2.3.4")
        _, tunggu2 = t.penalty("1.2.3.4")
        assert tunggu2 > tunggu1

    def test_penebak_terkunci_setelah_ambang(self) -> None:
        t = AuthThrottle(threshold=3, lockout_after=10)
        for _ in range(10):
            t.record_failure("1.2.3.4")
        terkunci, _ = t.penalty("1.2.3.4")
        assert terkunci

    def test_alamat_lain_tidak_ikut_terhukum(self) -> None:
        t = AuthThrottle(threshold=2, lockout_after=5)
        for _ in range(5):
            t.record_failure("1.2.3.4")
        assert t.penalty("9.9.9.9") == (False, 0)

    def test_berhasil_menghapus_catatan_gagal(self) -> None:
        """Salah pasang kunci sesaat tidak boleh mengunci integrasi selamanya."""
        t = AuthThrottle(threshold=2)
        t.record_failure("1.2.3.4")
        t.record_failure("1.2.3.4")
        t.record_success("1.2.3.4")
        assert t.penalty("1.2.3.4") == (False, 0)


# --- penyuntingan log & kripto ---------------------------------------------

class TestRedaction:
    def test_kunci_api_disunting(self) -> None:
        teks = "gagal memakai ragk_0123456789abcdef.rahasia-sekali"
        assert "rahasia-sekali" not in redact.scrub_text(teks)

    @pytest.mark.parametrize(
        "field", ["password", "api_key", "authorization", "secret_hash", "token"],
    )
    def test_field_sensitif_diganti(self, field: str) -> None:
        assert redact.scrub({field: "nilai-rahasia"})[field] == redact.MASK

    def test_student_id_disamarkan_tetapi_tetap_dapat_dicocokkan(self) -> None:
        """Cukup untuk menelusuri satu kasus, tidak cukup untuk mengenali orangnya."""
        hasil = redact.scrub({"student_id": "2021001234"})
        assert hasil["student_id"] != "2021001234"
        assert hasil["student_id"] == redact.scrub({"student_id": "2021001234"})["student_id"]

    def test_bekerja_rekursif(self) -> None:
        hasil = redact.scrub({"a": {"b": [{"password": "x"}]}})
        assert hasil["a"]["b"][0]["password"] == redact.MASK

    def test_struktur_terlalu_dalam_tidak_menggantung(self) -> None:
        dalam: dict = {}
        p = dalam
        for _ in range(50):
            p["x"] = {}
            p = p["x"]
        assert redact.scrub(dalam) is not None


class TestCrypto:
    def test_indeks_buta_deterministik_dan_terpisah_per_tenant(self) -> None:
        assert crypto.blind_index("2021001", tenant_id=A) == \
               crypto.blind_index("2021001", tenant_id=A)
        assert crypto.blind_index("2021001", tenant_id=A) != \
               crypto.blind_index("2021001", tenant_id=B)

    def test_indeks_buta_tidak_memuat_nilai_aslinya(self) -> None:
        assert "2021001" not in (crypto.blind_index("2021001", tenant_id=A) or "")

    def test_nilai_kosong_dilewati(self) -> None:
        assert crypto.blind_index("", tenant_id=A) is None
        assert crypto.encrypt_field(None) is None

    def test_penyandian_mati_membiarkan_nilai_apa_adanya(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.config import settings
        monkeypatch.setattr(settings, "encrypt_pii", False)
        assert crypto.encrypt_field("2021001", aad=A) == "2021001"

    def test_nilai_lama_belum_tersandi_tetap_terbaca(self) -> None:
        """Penyandian dapat dinyalakan tanpa migrasi serentak."""
        assert crypto.decrypt_field("2021001", aad=A) == "2021001"

    def test_bolak_balik_bila_tersedia(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("cryptography")
        import base64

        from pydantic import SecretStr

        from src.config import settings
        monkeypatch.setattr(
            settings, "pii_kek",
            SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode()),
        )
        monkeypatch.setattr(settings, "encrypt_pii", True)
        crypto.set_key_provider(None)

        sandi = crypto.encrypt_field("2021001", aad=A)
        assert sandi != "2021001"
        assert crypto.decrypt_field(sandi, aad=A) == "2021001"
        # Diikat pada tenant: sandi tenant A tidak terbaca sebagai milik B.
        assert crypto.decrypt_field(sandi, aad=B) is None
        crypto.set_key_provider(None)

    def test_nilai_sama_menghasilkan_sandi_berbeda(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sandi deterministik akan membocorkan siapa sama dengan siapa."""
        pytest.importorskip("cryptography")
        import base64

        from pydantic import SecretStr

        from src.config import settings
        monkeypatch.setattr(
            settings, "pii_kek",
            SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode()),
        )
        monkeypatch.setattr(settings, "encrypt_pii", True)
        crypto.set_key_provider(None)

        assert crypto.encrypt_field("2021001", aad=A) != \
               crypto.encrypt_field("2021001", aad=A)
        crypto.set_key_provider(None)
