"""Regresi untuk empat celah yang sempat ada di versi pertama migrasi multi-tenant.

Keempatnya lolos dari uji lapisan penyimpanan (`tests/test_tenancy.py`) karena
letaknya di lapisan **rute dan autentikasi** — tempat keputusan "boleh atau
tidak" sesungguhnya diambil. Penyimpanan sudah benar menolak `tenant_id` kosong;
yang salah adalah siapa yang boleh memanggilnya dan dengan cakupan apa.

Karena itu seluruh uji di sini menembak lewat HTTP dengan **autentikasi
sungguhan** — kunci API asli, bukan dependency yang ditimpa. Menimpa
`require_tenant` akan melewatkan persis kode yang sedang diuji.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from src.api.dependencies import get_pipeline
from src.api.main import app
from src.config import settings
from src.pipeline import RAGPipeline
from src.schemas import Chunk, ElementType
from src.storage import content_store, tenant_store
from src.storage.qdrant_store import QdrantStore
from src.tenancy import ALL_SCOPES

TENANT = "kampus-a"


def _chunk(label: str, *, course: str, source: str, teks: str) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid5(uuid.NAMESPACE_OID, label)),
        text=teks,
        parent_element_id="e1",
        element_type=ElementType.TEXT,
        source_file=source,
        content_id=f"{course}-minggu-1",
        course_id=course,
        course_name=course.upper(),
        week=1,
        dense_embedding=[0.1] * 1024,
    )


@pytest.fixture
def dunia(monkeypatch: pytest.MonkeyPatch):
    """Satu tenant, dua kunci (penuh & terbatas), dua mata kuliah terindeks.

    Kunci terbatas hanya boleh menyentuh `sbd`. Seluruh percobaan menembus ke
    `kka` di bawah ini adalah percobaan melewati batas itu.
    """
    monkeypatch.setattr(settings, "rate_limit_enabled", False, raising=False)

    tenant_store.create_tenant(TENANT, name="Kampus A")
    kunci_penuh = tenant_store.issue_key(TENANT, scopes=set(ALL_SCOPES)).raw
    kunci_terbatas = tenant_store.issue_key(
        TENANT, scopes=set(ALL_SCOPES), allowed_courses=["sbd"],
    ).raw

    store = QdrantStore(
        client=QdrantClient(":memory:"), collection="regresi", enable_sparse=False,
    )

    async def isi() -> None:
        await store.ensure_collection()
        await store.upsert_chunks(
            [_chunk("r1", course="sbd", source="sbd1.pdf", teks="Normalisasi")],
            tenant_id=TENANT,
        )
        await store.upsert_chunks(
            [_chunk("r2", course="kka", source="kka1.pdf", teks="Kalkulus")],
            tenant_id=TENANT,
        )

    asyncio.run(isi())

    class GeneratorPalsu:
        """LLM tiruan.

        Ada supaya uji kontrol positif — permintaan yang MEMANG boleh lolos —
        benar-benar sampai ke ujung dan membuktikan penjagaannya tidak kebablasan.
        Tanpa ini, "lolos dari penjagaan" tak terbedakan dari "gagal karena
        sebab lain".
        """

        async def generate_quiz(self, *a, **kw):
            return [{
                "question": "Apa itu normalisasi?",
                "options": ["A", "B", "C", "D"],
                "answer_index": 0,
                "explanation": "",
            }]

        async def generate_starter_questions(self, *a, **kw):
            return ["Apa inti materi ini?"]

    class PipelinePalsu(RAGPipeline):
        """Hanya membawa store; indexing sengaja gagal dengan pesan khas pustaka."""

        def __init__(self) -> None:
            self._store = store
            self._generator = GeneratorPalsu()

        async def index_document(self, *a, **kw):
            raise RuntimeError("psycopg2.OperationalError di /var/lib/rahasia")

    app.dependency_overrides[get_pipeline] = lambda: PipelinePalsu()
    yield TestClient(app), kunci_penuh, kunci_terbatas
    app.dependency_overrides.clear()


def _h(kunci: str) -> dict[str, str]:
    return {"X-API-Key": kunci}


# --- 1. perlambatan auth tidak boleh mengunci kunci yang sah ----------------

class TestPerlambatanAuth:
    def test_kunci_sah_lolos_walau_alamatnya_sedang_dihukum(self, dunia) -> None:
        """Hukuman perlambatan berlaku bagi yang GAGAL, bukan bagi alamatnya.

        Versi pertama menolak sebelum kunci diverifikasi. Akibatnya kunci sah pun
        ikut ditolak — dan karena penolakan terjadi sebelum verifikasi, tidak ada
        lagi jalan mencatat keberhasilan yang menghapus hukumannya: alamat itu
        terkunci penuh 15 menit. Di balik NAT kampus, satu klien yang salah
        pasang kunci akan mematikan seluruh klien sah di gedung yang sama.
        """
        c, kunci_penuh, _ = dunia
        for _ in range(12):
            c.get("/catalog/courses", headers=_h("ragk_0123456789abcdef.salah"))

        r = c.get("/catalog/courses", headers=_h(kunci_penuh))
        assert r.status_code == 200, (
            f"kunci sah ikut terkunci ({r.status_code}) — pelanggan sah tak bisa masuk"
        )

    def test_penebak_tetap_diperlambat(self, dunia) -> None:
        """Menunda hukuman tidak boleh berarti menghapusnya."""
        c, _, _ = dunia
        kode = [
            c.get("/catalog/courses", headers=_h("ragk_0123456789abcdef.salah")).status_code
            for _ in range(12)
        ]
        assert 429 in kode, "penebakan berulang tidak pernah diperlambat"

    def test_keberhasilan_menghapus_catatan_kegagalan(self, dunia) -> None:
        """Setelah kunci sah dipakai, penghitung alamat itu harus bersih lagi."""
        c, kunci_penuh, _ = dunia
        for _ in range(12):
            c.get("/catalog/courses", headers=_h("ragk_0123456789abcdef.salah"))
        c.get("/catalog/courses", headers=_h(kunci_penuh))
        # Satu kegagalan sesudahnya belum melewati ambang, jadi 401 — bukan 429.
        r = c.get("/catalog/courses", headers=_h("ragk_0123456789abcdef.salah"))
        assert r.status_code == 401


# --- 2. ABAC pada cakupan pencarian yang sebenarnya -------------------------

class TestAbacCakupanPencarian:
    def test_kunci_terbatas_tidak_boleh_mencari_tanpa_mata_kuliah(self, dunia) -> None:
        """Tanpa mata kuliah, pencarian menyapu SELURUH materi tenant.

        Versi pertama hanya memeriksa ketika klien mengirim `course_id` — jadi
        pembatasannya justru dapat dilewati dengan TIDAK mengirimkannya.
        """
        c, _, kunci_terbatas = dunia
        r = c.post(
            "/chat/ask",
            headers=_h(kunci_terbatas),
            json={"question": "jelaskan integral lipat dua", "guided": False},
        )
        assert r.status_code == 403, (
            f"kunci terbatas menyapu seluruh materi tenant (status {r.status_code})"
        )

    def test_kunci_terbatas_ditolak_saat_menyebut_mata_kuliah_lain(self, dunia) -> None:
        c, _, kunci_terbatas = dunia
        r = c.post(
            "/chat/ask",
            headers=_h(kunci_terbatas),
            json={"question": "apa itu limit", "course_id": "kka", "guided": False},
        )
        assert r.status_code == 403

    def test_kunci_penuh_tidak_terpengaruh(self, dunia) -> None:
        """Kunci tanpa pembatasan tidak boleh ikut terkena aturan itu."""
        c, kunci_penuh, _ = dunia
        r = c.get("/catalog/courses", headers=_h(kunci_penuh))
        assert r.status_code == 200
        assert {x["course_id"] for x in r.json()["courses"]} == {"sbd", "kka"}

    def test_katalog_hanya_menawarkan_mata_kuliah_jatahnya(self, dunia) -> None:
        c, _, kunci_terbatas = dunia
        r = c.get("/catalog/courses", headers=_h(kunci_terbatas))
        assert [x["course_id"] for x in r.json()["courses"]] == ["sbd"]

    def test_minggu_mata_kuliah_lain_ditolak(self, dunia) -> None:
        c, _, kunci_terbatas = dunia
        assert c.get(
            "/catalog/courses/kka/weeks", headers=_h(kunci_terbatas),
        ).status_code == 403


# --- 3. ABAC pada endpoint yang hanya menerima content_id -------------------

def _jalur(content_id: str) -> list[tuple[str, str, dict | None]]:
    """Kesembilan titik masuk yang dialamati lewat `content_id`, untuk satu materi.

    Disusun sebagai fungsi, bukan daftar tetap, supaya uji penolakan dan uji
    kontrol positif menembak jalur yang BENAR-BENAR sama dan hanya materinya yang
    berbeda. Dua daftar terpisah akan menyimpang diam-diam begitu ada endpoint
    baru, dan yang menyimpang justru sisi yang membuktikan penjagaannya bekerja.
    """
    berkas = f"{content_id.split('-')[0]}1.pdf"
    return [
        ("get", f"/catalog/materials/{content_id}/{berkas}/quiz", None),
        ("get", f"/catalog/materials/{content_id}/{berkas}/starter-questions", None),
        ("post", f"/catalog/materials/{content_id}/{berkas}/quiz/submit",
         {"answers": [0]}),
        ("get", f"/guided/topic-hints?content_id={content_id}", None),
        ("post", "/guided/ask",
         {"template_id": "understand_concept", "topic": "x", "content_id": content_id}),
        ("post", "/guided/ask/stream",
         {"template_id": "understand_concept", "topic": "x", "content_id": content_id}),
        ("post", "/guided/exercise", {"topic": "x", "content_id": content_id}),
        ("get", f"/concepts/preview?content_id={content_id}", None),
        ("post", "/concepts/questions",
         {"content_id": content_id, "concepts": [{"nama": "Konsep"}]}),
    ]


class TestAbacLewatContentId:
    """Endpoint ini tidak menerima `course_id`, sehingga sempat terlewat.

    Kesembilannya membangkitkan keluaran dari ISI materi. Membiarkannya terbuka
    membuat pembatasan pada endpoint ber-`course_id` kehilangan artinya: jatah
    kunci dapat dilewati lewat jalan memutar — ditolak 403 di `/chat/ask`, tetapi
    dilayani di `/guided/ask` untuk materi yang persis sama.
    """

    @pytest.mark.parametrize("metode, path, body", _jalur("kka-minggu-1"))
    def test_lintas_mata_kuliah_ditolak(self, dunia, metode, path, body) -> None:
        c, _, kunci_terbatas = dunia
        kirim = getattr(c, metode)
        r = kirim(path, headers=_h(kunci_terbatas), **({"json": body} if body else {}))
        assert r.status_code == 403, (
            f"{path} bocor ke kunci terbatas (status {r.status_code})"
        )

    @pytest.mark.parametrize("metode, path, body", _jalur("sbd-minggu-1"))
    def test_mata_kuliah_sendiri_tidak_ikut_tertutup(
        self, dunia, metode, path, body,
    ) -> None:
        """Penjagaan tidak boleh kebablasan menutup jatahnya sendiri.

        Yang diuji hanya "penjagaan tidak berbunyi", bukan jawabannya: pipeline
        tiruan di sini memang tidak bisa menjawab, sehingga sebagian jalur
        berakhir 5xx. Itu cukup — 403 hanya mungkin datang dari penjagaan ini.
        """
        c, _, kunci_terbatas = dunia
        kirim = getattr(c, metode)
        r = kirim(path, headers=_h(kunci_terbatas), **({"json": body} if body else {}))
        assert r.status_code != 403, f"{path} menutup jatah kuncinya sendiri"

    @pytest.mark.parametrize(
        "path, body",
        [
            ("/guided/ask", {"template_id": "understand_concept", "topic": "x"}),
            ("/guided/ask/stream", {"template_id": "understand_concept", "topic": "x"}),
            ("/guided/exercise", {"topic": "x"}),
        ],
    )
    def test_tanpa_content_id_ditolak(self, dunia, path, body) -> None:
        """`content_id` boleh kosong di ketiga rute ini — dan itu jalan memutarnya.

        Tanpa materi tertentu, kueri di hilir mencakup seluruh milik tenant.
        Versi pertama penjagaan ini menyerahkan `None` ke `get_content_course`,
        yang memulangkan mata kuliah titik mana pun yang kebetulan terbaca lebih
        dulu — sehingga kunci terbatas lolos atau ditolak secara untung-untungan,
        bergantung pada urutan titik di dalam indeks.
        """
        c, _, kunci_terbatas = dunia
        r = c.post(path, headers=_h(kunci_terbatas), json=body)
        assert r.status_code == 403, (
            f"{path} tanpa content_id menyapu seluruh materi tenant "
            f"(status {r.status_code})"
        )

    def test_materi_tak_dikenal_ditolak(self, dunia) -> None:
        """Materi yang tidak ada pun ditolak, bukan diteruskan.

        Membedakan "belum diunggah" dari "milik fakultas lain" adalah keterangan
        yang justru tidak berhak diketahui kunci terbatas.
        """
        c, _, kunci_terbatas = dunia
        r = c.get(
            "/guided/topic-hints?content_id=hantu-minggu-9", headers=_h(kunci_terbatas),
        )
        assert r.status_code == 403

    def test_kunci_penuh_tidak_ikut_terjaga(self, dunia) -> None:
        """Kunci tanpa pembatasan tidak boleh membayar kueri tambahan ini."""
        c, kunci_penuh, _ = dunia
        r = c.get(
            "/guided/topic-hints?content_id=kka-minggu-1", headers=_h(kunci_penuh),
        )
        assert r.status_code == 200


# --- 4. daftar galat batch tidak boleh memantulkan teks pustaka -------------

class TestKebocoranLewatBodySukses:
    def test_galat_batch_tidak_memuat_rincian_internal(self, dunia) -> None:
        """`errors` terkirim sebagai body 200, jadi ia melewati penjagaan 5xx.

        Kebocoran yang sama seperti pada jalur unggah dan chat, hanya lewat pintu
        berbeda — dan lebih mudah terlewat justru karena balasannya sukses.
        """
        c, kunci_penuh, _ = dunia
        d = content_store.storage_root(tenant_id=TENANT) / "sbd-minggu-1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "bab1.pdf").write_text("isi", encoding="utf-8")

        r = c.post(
            "/documents/index-batch",
            headers=_h(kunci_penuh),
            json={"content_id": "sbd-minggu-1"},
        )
        assert r.status_code == 200
        galat = " ".join(r.json()["errors"])
        assert galat, "seharusnya ada satu kegagalan tercatat"
        assert "psycopg2" not in galat and "/var/lib/rahasia" not in galat, (
            f"rincian internal bocor lewat body sukses: {galat}"
        )


# --- 5. penjagaan materi tidak boleh bertanya ke indeks tanpa materi --------

class TestGuardMaterialTanpaMateri:
    """Mengapa diuji langsung, bukan lewat HTTP.

    Lewat HTTP, kasus `content_id` kosong lulus secara KEBETULAN: pemindaian
    tanpa saringan materi memulangkan titik mana pun yang terbaca lebih dulu,
    dan di fixture itu kebetulan mata kuliah yang memang terlarang. Ganti urutan
    unggahnya dan uji yang sama lulus padahal celahnya terbuka lebar.

    Yang harus dipaku adalah mekanismenya: tanpa materi tertentu, penjagaan
    menolak TANPA menanyakan apa pun ke indeks.
    """

    class _StorePencatat:
        def __init__(self) -> None:
            self.ditanya: list[str | None] = []

        async def get_content_course(self, content_id, *, tenant_id):  # noqa: ANN001
            self.ditanya.append(content_id)
            return "sbd"  # jawaban yang akan MELOLOSKAN kunci terbatas

    class _PipelineTiruan:
        def __init__(self, store) -> None:  # noqa: ANN001
            self._store = store

    @pytest.mark.parametrize("kosong", [None, ""])
    def test_tanpa_materi_ditolak_tanpa_menyentuh_indeks(self, kosong) -> None:
        from src.api.dependencies import guard_material
        from src.tenancy import TenantContext, TenantPermissionError

        store = self._StorePencatat()
        tenant = TenantContext(tenant_id=TENANT, allowed_courses=frozenset({"sbd"}))

        with pytest.raises(TenantPermissionError):
            asyncio.run(guard_material(tenant, self._PipelineTiruan(store), kosong))

        assert store.ditanya == [], (
            "indeks tetap ditanya tanpa materi — jawabannya mata kuliah sembarang, "
            f"jadi lolosnya untung-untungan (dikonsultasi dengan {store.ditanya})"
        )

    def test_kunci_tanpa_batasan_tetap_lolos(self) -> None:
        """Penolakan itu hanya berlaku bagi kunci yang memang dibatasi."""
        from src.api.dependencies import guard_material
        from src.tenancy import TenantContext

        store = self._StorePencatat()
        tenant = TenantContext(tenant_id=TENANT, allowed_courses=None)
        asyncio.run(guard_material(tenant, self._PipelineTiruan(store), None))
        assert store.ditanya == [], "kunci tanpa batasan ikut membayar kueri tambahan"

    def test_materi_tak_dikenal_ditolak(self) -> None:
        """`get_content_course` memulangkan None bila materinya tidak ada."""
        from src.api.dependencies import guard_material
        from src.tenancy import TenantContext, TenantPermissionError

        class _Kosong:
            async def get_content_course(self, content_id, *, tenant_id):  # noqa: ANN001
                return None

        tenant = TenantContext(tenant_id=TENANT, allowed_courses=frozenset({"sbd"}))
        with pytest.raises(TenantPermissionError):
            asyncio.run(
                guard_material(tenant, self._PipelineTiruan(_Kosong()), "hantu-minggu-9"),
            )
