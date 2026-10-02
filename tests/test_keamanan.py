"""Lapisan keamanan: jejak audit, penyuntingan log, penyandian PII, ABAC.

Isolasi antar kampus diuji di `tests/test_tenancy.py`. Berkas ini menguji
pengaman yang MENYERTAINYA — yang bekerja setelah sebuah permintaan sudah tahu
milik kampus mana, dan yang menentukan apakah jejaknya dapat dipercaya:

- `security/audit.py`  — rantai HMAC: perubahan dan penghapusan baris terdeteksi.
- `security/redact.py` — kunci dan identitas mahasiswa tidak pernah utuh di log.
- `security/crypto.py` — `student_id` tersandi, tetapi tetap dapat disaring.
- `security/ratelimit.AuthThrottle` — penebak kunci diperlambat, yang sah tidak.
- `tenancy.TenantContext` — RBAC (`scopes`) dan ABAC (`allowed_courses`).
"""
from __future__ import annotations

import base64

import pytest
from pydantic import SecretStr

from src.config import settings
from src.security import audit, crypto, redact
from src.security.ratelimit import AuthThrottle, MemoryBackend, TenantRateLimiter
from src.tenancy import (
    DEFAULT_KEY_SCOPES,
    LEGACY_FULL_SCOPES,
    READER_SCOPES,
    SCOPE_ADMIN,
    SCOPE_CHAT,
    SCOPE_CONTENT_WRITE,
    TenantContext,
    TenantPermissionError,
    TenantScopeError,
    cache_key,
)

A = "kampus-a"
B = "kampus-b"


# --- RBAC & ABAC ------------------------------------------------------------

class TestKewenanganKunci:
    def test_konteks_menolak_tenant_tak_sah(self) -> None:
        with pytest.raises(TenantScopeError):
            TenantContext(tenant_id="../lain")

    def test_hak_akses_ditegakkan(self) -> None:
        pembaca = TenantContext(tenant_id=A, scopes=READER_SCOPES)
        assert pembaca.has(SCOPE_CHAT)
        assert not pembaca.has(SCOPE_CONTENT_WRITE)
        with pytest.raises(TenantPermissionError):
            pembaca.require(SCOPE_CONTENT_WRITE)

    def test_kunci_pembaca_tidak_pernah_ber_hak_admin(self) -> None:
        assert SCOPE_ADMIN not in READER_SCOPES

    def test_abac_membatasi_mata_kuliah(self) -> None:
        t = TenantContext(tenant_id=A, allowed_courses=frozenset({"sbd"}))
        assert t.may_access_course("sbd")
        assert not t.may_access_course("kka")
        with pytest.raises(TenantPermissionError):
            t.require_course("kka")

    def test_abac_menolak_kueri_tanpa_mata_kuliah(self) -> None:
        """Tanpa mata kuliah, kueri mencakup SELURUH milik kampus.

        Itu lebih luas daripada jatah kunci yang dibatasi, jadi harus ditolak —
        bukan diloloskan karena "tidak menyebut mata kuliah terlarang". Persis di
        sinilah pembatasan per mata kuliah paling mudah dilewati: dengan TIDAK
        mengirimkan `course_id` sama sekali.
        """
        t = TenantContext(tenant_id=A, allowed_courses=frozenset({"sbd"}))
        assert not t.may_access_course(None)

    def test_tanpa_batasan_boleh_semua(self) -> None:
        assert TenantContext(tenant_id=A).may_access_course("apa saja")

    def test_bentuk_untuk_log_tidak_memuat_kunci(self) -> None:
        t = TenantContext(tenant_id=A, key_id="abc123")
        assert t.redacted() == {"tenant_id": A, "key_id": "abc123"}

    def test_cache_key_selalu_diawali_kampus(self) -> None:
        """Tanpa ini, kuis dari materi kampus A tersaji ke kampus B ber-content_id sama."""
        assert cache_key(A, "quiz", "bab1.pdf") != cache_key(B, "quiz", "bab1.pdf")

    def test_cache_key_menolak_kampus_kosong(self) -> None:
        with pytest.raises(TenantScopeError):
            cache_key("", "quiz")


# --- jejak audit ------------------------------------------------------------

class TestRantaiAudit:
    """AUDIT_DIR dan `_tail_cache` sudah dialihkan ke tmp_path oleh conftest."""

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

    def test_jejak_terpisah_per_kampus(self) -> None:
        audit.record(audit.CHAT_ASK, tenant_id=A)
        audit.record(audit.CHAT_ASK, tenant_id=B)
        assert (audit.AUDIT_DIR / A).is_dir()
        assert (audit.AUDIT_DIR / B).is_dir()

    def test_kunci_tidak_pernah_masuk_jejak(self) -> None:
        audit.record(
            audit.AUTH_FAILURE, tenant_id=A,
            detail={"api_key": "ragk_0123456789abcdef.sangat-rahasia"},
        )
        isi = next((audit.AUDIT_DIR / A).glob("*.jsonl")).read_text(encoding="utf-8")
        assert "sangat-rahasia" not in isi

    def test_peristiwa_tanpa_kampus_tetap_tercatat(self) -> None:
        """Autentikasi gagal belum punya kampus, tetapi justru paling perlu dicatat."""
        audit.record(audit.AUTH_FAILURE, ip="1.2.3.4")
        assert (audit.AUDIT_DIR / audit.SYSTEM_SCOPE).is_dir()

    def test_gagal_menulis_tidak_melempar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unggahan yang sudah berhasil tidak boleh batal karena pencatatannya gagal."""
        monkeypatch.setattr(audit, "AUDIT_DIR", audit.AUDIT_DIR / "x\x00y")
        audit.record(audit.CHAT_ASK, tenant_id=A)   # tidak melempar apa pun


# --- perlambatan penebakan kunci -------------------------------------------

class TestPerlambatanAuth:
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
        """Di balik NAT kampus, satu klien salah pasang tidak boleh mematikan lainnya."""
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


class TestKuotaPerIp:
    async def test_batas_per_alamat_ditegakkan(self) -> None:
        lim = TenantRateLimiter(MemoryBackend())
        for _ in range(3):
            assert (await lim.check_ip("1.1.1.1", per_minute=3)).allowed
        putusan = await lim.check_ip("1.1.1.1", per_minute=3)
        assert not putusan.allowed and putusan.retry_after >= 1
        assert putusan.scope == "ip"

    async def test_alamat_lain_punya_jatah_sendiri(self) -> None:
        lim = TenantRateLimiter(MemoryBackend())
        for _ in range(3):
            await lim.check_ip("1.1.1.1", per_minute=3)
        assert (await lim.check_ip("2.2.2.2", per_minute=3)).allowed

    async def test_batas_nol_berarti_tanpa_batas(self) -> None:
        lim = TenantRateLimiter(MemoryBackend())
        for _ in range(50):
            assert (await lim.check_ip("1.1.1.1", per_minute=0)).allowed


# --- penyuntingan log -------------------------------------------------------

class TestPenyuntingan:
    def test_kunci_api_disunting(self) -> None:
        teks = "gagal memakai ragk_0123456789abcdef.rahasia-sekali"
        assert "rahasia-sekali" not in redact.scrub_text(teks)

    @pytest.mark.parametrize(
        "medan", ["password", "api_key", "authorization", "secret_hash", "token"],
    )
    def test_field_sensitif_diganti(self, medan: str) -> None:
        assert redact.scrub({medan: "nilai-rahasia"})[medan] == redact.MASK

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


# --- penyandian PII ---------------------------------------------------------

@pytest.fixture
def kek(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nyalakan penyandian dengan KEK dari variabel lingkungan (hanya untuk uji)."""
    pytest.importorskip("cryptography")
    monkeypatch.setattr(
        settings, "pii_kek",
        SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode()), raising=False,
    )
    monkeypatch.setattr(settings, "encrypt_pii", True, raising=False)
    crypto.set_key_provider(None)
    yield
    crypto.set_key_provider(None)


class TestKripto:
    def test_indeks_buta_deterministik_dan_terpisah_per_kampus(self) -> None:
        """Deterministik supaya bisa disaring; terpisah supaya tidak bisa dibandingkan.

        Tanpa `tenant_id` di dalam HMAC, dua kampus dapat menyimpulkan bahwa
        seseorang terdaftar di keduanya hanya dengan membandingkan indeks.
        """
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
        monkeypatch.setattr(settings, "encrypt_pii", False, raising=False)
        assert crypto.encrypt_field("2021001", aad=A) == "2021001"

    def test_nilai_lama_belum_tersandi_tetap_terbaca(self) -> None:
        """Penyandian dapat dinyalakan pada data yang ada tanpa migrasi serentak."""
        assert crypto.decrypt_field("2021001", aad=A) == "2021001"

    def test_bolak_balik_dan_terikat_pada_kampus(self, kek: None) -> None:
        sandi = crypto.encrypt_field("2021001", aad=A)
        assert sandi != "2021001"
        assert crypto.decrypt_field(sandi, aad=A) == "2021001"
        # Sandi milik kampus A yang dipindahkan ke berkas kampus B gagal
        # didekripsi, bukan diam-diam terbaca.
        assert crypto.decrypt_field(sandi, aad=B) is None

    def test_nilai_sama_menghasilkan_sandi_berbeda(self, kek: None) -> None:
        """Sandi deterministik akan membocorkan siapa sama dengan siapa."""
        assert crypto.encrypt_field("2021001", aad=A) != \
               crypto.encrypt_field("2021001", aad=A)

    def test_tersedia_saat_kek_terpasang(self, kek: None) -> None:
        assert crypto.is_available()


# --- registri kampus: hak warisan & kalender evaluasi -----------------------

class TestHakWarisan:
    """Kunci yang sudah beredar tidak boleh mendadak kehilangan akses.

    Hak `catalog:read`, `conversation:delete`, dan `answer:validate` baru ada
    setelah katalog, pengelolaan percakapan, dan validasi dosen digabungkan.
    Berkas registri milik kampus yang sudah berjalan hanya menyimpan empat nama
    hak lama.
    """

    def test_kunci_lama_berhak_penuh_ikut_naik(self) -> None:
        from src.storage import tenant_store

        rec = tenant_store.Tenant.from_dict({
            "tenant_id": "kampus-lama",
            "keys": [{
                "key_id": "a" * 16, "secret_hash": "x",
                "scopes": sorted(LEGACY_FULL_SCOPES),
            }],
        })
        assert set(rec.api_keys[0].scopes) == set(DEFAULT_KEY_SCOPES)
        assert SCOPE_ADMIN not in rec.api_keys[0].scopes

    def test_kunci_lama_bersempit_tidak_ikut_naik(self) -> None:
        """Kunci yang memang dibatasi harus tetap dibatasi."""
        from src.storage import tenant_store

        rec = tenant_store.Tenant.from_dict({
            "tenant_id": "kampus-lama",
            "keys": [{"key_id": "b" * 16, "secret_hash": "x", "scopes": [SCOPE_CHAT]}],
        })
        assert rec.api_keys[0].scopes == [SCOPE_CHAT]


class TestJadwalEvaluasiKampus:
    def test_jadwal_tersimpan_dan_ikut_ke_konteks(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Jadwal dibawa bersama identitas pemanggil, bukan dibaca ulang tiap rute."""
        from src import evaluation
        from src.storage import tenant_store

        monkeypatch.setattr(
            settings, "tenant_key_pepper", SecretStr("p" * 40), raising=False,
        )
        tenant_store.create_tenant("kampus-j", name="Kampus J")
        mentah = tenant_store.issue_key("kampus-j").raw
        assert tenant_store.authenticate(mentah).evaluation_schedule is None

        jadwal = evaluation.EvaluationSchedule.from_json(
            [{"week": 7, "kind": "ets", "covers": [1, 7]}]
        )
        tenant_store.set_schedule("kampus-j", jadwal)
        ctx = tenant_store.authenticate(mentah)
        assert ctx is not None and ctx.evaluation_schedule is not None
        assert ctx.evaluation_schedule.weeks == (7,)

        tenant_store.set_schedule("kampus-j", None)
        assert tenant_store.authenticate(mentah).evaluation_schedule is None


class TestTenantctlLanjutan:
    def test_schedule_set_lihat_reset(self, capsys: pytest.CaptureFixture) -> None:
        from scripts import tenantctl
        from src.storage import tenant_store

        tenant_store.create_tenant("kampus-k")
        assert tenantctl.main(["schedule", "kampus-k", "--set", "7:ets:1-7"]) == 0
        assert tenant_store.get("kampus-k").evaluation_schedule.weeks == (7,)

        capsys.readouterr()
        assert tenantctl.main(["schedule", "kampus-k"]) == 0
        assert "sendiri" in capsys.readouterr().out

        assert tenantctl.main(["schedule", "kampus-k", "--reset"]) == 0
        assert tenant_store.get("kampus-k").evaluation_schedule is None

    def test_schedule_cacat_ditolak_tanpa_menulis(
        self, capsys: pytest.CaptureFixture,
    ) -> None:
        """Jadwal cacat harus berhenti di CLI, bukan tersimpan lalu dipakai salah."""
        from scripts import tenantctl
        from src.storage import tenant_store

        tenant_store.create_tenant("kampus-l")
        assert tenantctl.main(["schedule", "kampus-l", "--set", "tujuh:ets:1-7"]) == 1
        assert "tidak sah" in capsys.readouterr().err
        assert tenant_store.get("kampus-l").evaluation_schedule is None

    def test_verify_audit_melaporkan_rantai_putus(
        self, capsys: pytest.CaptureFixture,
    ) -> None:
        from scripts import tenantctl

        audit.record(audit.CHAT_ASK, tenant_id="kampus-m", detail={"n": 1})
        audit.record(audit.CHAT_ASK, tenant_id="kampus-m", detail={"n": 2})
        assert tenantctl.main(["verify-audit", "kampus-m"]) == 0

        berkas = next((audit.AUDIT_DIR / "kampus-m").glob("*.jsonl"))
        baris = berkas.read_text(encoding="utf-8").splitlines()
        berkas.write_text("\n".join(baris[:1]) + "\n", encoding="utf-8")
        assert tenantctl.main(["verify-audit", "kampus-m"]) == 0  # potong di ekor: utuh

        berkas.write_text(
            baris[0].replace('"n": 1', '"n": 9') + "\n" + baris[1] + "\n",
            encoding="utf-8",
        )
        assert tenantctl.main(["verify-audit", "kampus-m"]) == 1
        assert "GAGAL" in capsys.readouterr().out

    def test_kunci_admin_hanya_lewat_flag(self, capsys: pytest.CaptureFixture) -> None:
        from scripts import tenantctl
        from src.storage import tenant_store

        tenant_store.create_tenant("kampus-n")
        assert tenantctl.main(["issue-key", "kampus-n"]) == 0
        assert SCOPE_ADMIN not in tenant_store.get("kampus-n").api_keys[0].scopes

        assert tenantctl.main(["issue-key", "kampus-n", "--admin"]) == 0
        assert SCOPE_ADMIN in tenant_store.get("kampus-n").api_keys[1].scopes

    def test_courses_membatasi_kunci(self) -> None:
        from scripts import tenantctl
        from src.storage import tenant_store

        tenant_store.create_tenant("kampus-o")
        assert tenantctl.main(["issue-key", "kampus-o", "--courses", "sbd, kka"]) == 0
        assert tenant_store.get("kampus-o").api_keys[0].allowed_courses == ["kka", "sbd"]


# --- perlambatan auth lewat HTTP sungguhan ---------------------------------

@pytest.fixture
def klien_auth(monkeypatch: pytest.MonkeyPatch):
    """Satu rute yang dijaga `require_tenant`, dipanggil dengan kunci sungguhan.

    Sengaja TIDAK memakai penimpaan dependency: yang diuji justru kode di dalam
    `require_tenant`, dan menimpanya berarti melewati persis bagian itu.
    """
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    from src.api import errors
    from src.api.auth import require_tenant
    from src.storage import tenant_store

    monkeypatch.setattr(
        settings, "tenant_key_pepper", SecretStr("p" * 40), raising=False,
    )
    app = FastAPI()
    errors.install(app)

    @app.get("/uji", dependencies=[Depends(require_tenant)])
    async def _uji() -> dict[str, str]:
        return {"ok": "ya"}

    tenant_store.create_tenant("kampus-p", name="Kampus P")
    mentah = tenant_store.issue_key("kampus-p").raw
    return TestClient(app), mentah


_KUNCI_SALAH = {"X-API-Key": "ragk_0123456789abcdef.salah"}


class TestPerlambatanLewatHttp:
    def test_kunci_sah_lolos_walau_alamatnya_sedang_dihukum(self, klien_auth) -> None:
        """Hukuman berlaku bagi yang GAGAL, bukan bagi alamatnya.

        Versi pertama menolak SEBELUM kunci diverifikasi. Akibatnya kunci sah pun
        ikut ditolak — dan karena penolakan terjadi sebelum verifikasi, tidak ada
        lagi jalan mencatat keberhasilan yang menghapus hukumannya: alamat itu
        terkunci penuh 15 menit. Di balik NAT kampus, satu klien yang salah pasang
        kunci akan mematikan seluruh klien sah di gedung yang sama.
        """
        c, kunci = klien_auth
        for _ in range(12):
            c.get("/uji", headers=_KUNCI_SALAH)
        r = c.get("/uji", headers={"X-API-Key": kunci})
        assert r.status_code == 200, "kunci sah ikut terkunci — pelanggan sah tak bisa masuk"

    def test_penebak_tetap_diperlambat(self, klien_auth) -> None:
        """Menunda hukuman tidak boleh berarti menghapusnya."""
        c, _ = klien_auth
        kode = [c.get("/uji", headers=_KUNCI_SALAH).status_code for _ in range(12)]
        assert 429 in kode, "penebakan berulang tidak pernah diperlambat"

    def test_keberhasilan_menghapus_catatan_kegagalan(self, klien_auth) -> None:
        c, kunci = klien_auth
        for _ in range(12):
            c.get("/uji", headers=_KUNCI_SALAH)
        c.get("/uji", headers={"X-API-Key": kunci})
        # Satu kegagalan sesudahnya belum melewati ambang, jadi 401 — bukan 429.
        assert c.get("/uji", headers=_KUNCI_SALAH).status_code == 401

    def test_kuota_per_alamat_menutup_pemanggil_tanpa_kunci(
        self, klien_auth, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lapis per kampus tak menyentuh permintaan yang tak membawa kunci."""
        monkeypatch.setattr(settings, "rate_limit_ip_per_minute", 3, raising=False)
        c, _ = klien_auth
        kode = [c.get("/uji").status_code for _ in range(6)]
        assert 429 in kode

    def test_autentikasi_masuk_jejak_audit(self, klien_auth) -> None:
        c, kunci = klien_auth
        c.get("/uji", headers={"X-API-Key": kunci})
        berkas = next((audit.AUDIT_DIR / "kampus-p").glob("*.jsonl"))
        isi = berkas.read_text(encoding="utf-8")
        assert audit.AUTH_SUCCESS in isi
        assert kunci.split(".", 1)[1] not in isi   # rahasianya tidak pernah ikut
