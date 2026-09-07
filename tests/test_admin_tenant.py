"""Pengelolaan tenant lewat HTTP.

Endpoint paling berbahaya di layanan ini: ia dapat menerbitkan kunci untuk
tenant MANA PUN. Kebocorannya membuka seluruh pelanggan sekaligus, bukan satu.
Karena itu yang diuji di sini bukan hanya "fiturnya jalan", tetapi juga
"pengamannya benar-benar menutup".
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.storage import tenant_store
from src.tenancy import ALL_SCOPES, READER_SCOPES, SCOPE_ADMIN


@pytest.fixture
def admin_app(monkeypatch):
    """Aplikasi berisi router admin + katalog, tanpa menjalankan lifespan.

    Dibangun langsung alih-alih memuat ulang `src.api.main`: memuat ulang modul
    itu membuat objek `app` baru sehingga penimpaan dependency milik conftest
    ikut hilang, dan `with TestClient(...)` akan menjalankan lifespan yang
    memuat model 2 GB — mahal dan tidak ada hubungannya dengan yang diuji.
    Syarat `enable_admin_api` sendiri diuji terpisah di TestPengamanBawaan.
    """
    from fastapi import Depends, FastAPI

    from src.api import errors
    from src.api.auth import require_tenant
    from src.api.routes import admin, catalog
    from src.config import settings

    monkeypatch.setattr(settings, "rate_limit_enabled", False, raising=False)
    monkeypatch.setattr(
        settings, "tenant_key_pepper",
        SecretStr("pepper-uji-yang-cukup-panjang"), raising=False,
    )

    app = FastAPI()
    errors.install(app)
    auth = [Depends(require_tenant)]
    app.include_router(admin.router, dependencies=auth)
    app.include_router(catalog.router, dependencies=auth)
    app.state.pipeline = None
    return app


@pytest.fixture
def admin_key(admin_app):
    """Tenant operator beserta kunci ber-hak admin."""
    tenant_store.create_tenant("operator", "Operator Layanan")
    return tenant_store.issue_key(
        "operator", label="admin", scopes=set(ALL_SCOPES),
    ).raw


def _h(kunci: str) -> dict[str, str]:
    return {"X-API-Key": kunci}


class TestPengamanBawaan:
    def test_admin_mati_secara_bawaan(self) -> None:
        # Tanpa dinyalakan eksplisit, rutenya tidak boleh ada sama sekali —
        # bukan sekadar menolak, melainkan memang tidak terdaftar.
        from src.api.main import app

        assert not [r for r in app.routes if r.path.startswith("/admin")]

    def test_hak_admin_bukan_bagian_kunci_mahasiswa(self) -> None:
        assert SCOPE_ADMIN not in READER_SCOPES

    def test_tenant_dev_tidak_ber_hak_admin(self) -> None:
        # Tenant dadakan di pengembangan lahir tanpa kredensial apa pun.
        # Memberinya hak menerbitkan kunci berarti siapa pun yang menjangkau
        # port ini dapat membuat kunci untuk tenant mana pun.
        from src.api.auth import _dev_tenant

        assert not _dev_tenant().has(SCOPE_ADMIN)


class TestOtorisasi:
    def test_tanpa_kunci_ditolak(self, admin_app) -> None:
        # Di pengembangan tanpa tenant terdaftar, permintaan tanpa kunci
        # dilayani sebagai tenant "dev" — tetapi tenant itu tidak ber-hak
        # admin, sehingga tetap ditolak di sini.
        c = TestClient(admin_app)
        assert c.get("/admin/tenants").status_code in (401, 403)

    def test_kunci_mahasiswa_ditolak(self, admin_app, admin_key) -> None:
        tenant_store.create_tenant("kampus-a", "Kampus A")
        kunci_mhs = tenant_store.issue_key("kampus-a", scopes=set(READER_SCOPES)).raw
        c = TestClient(admin_app)
        assert c.get("/admin/tenants", headers=_h(kunci_mhs)).status_code == 403

    def test_kunci_admin_diterima(self, admin_app, admin_key) -> None:
        c = TestClient(admin_app)
        assert c.get("/admin/tenants", headers=_h(admin_key)).status_code == 200


class TestSiklusHidupTenant:
    def test_buat_lalu_terbitkan_kunci(self, admin_app, admin_key) -> None:
        c, h = TestClient(admin_app), _h(admin_key)

        r = c.post(
            "/admin/tenants",
            json={"tenant_id": "kampus-b", "name": "Kampus B"}, headers=h,
        )
        assert r.status_code == 201
        assert r.json()["tenant_id"] == "kampus-b"

        r = c.post(
            "/admin/tenants/kampus-b/keys",
            json={"label": "aplikasi mahasiswa"}, headers=h,
        )
        assert r.status_code == 201
        body = r.json()
        # Kunci mentah hanya muncul di sini, sekali seumur hidupnya.
        assert body["api_key"].startswith("ragk_")
        assert "tidak akan ditampilkan lagi" in body["peringatan"]

    def test_kunci_mentah_tidak_muncul_lagi_di_rincian(
        self, admin_app, admin_key,
    ) -> None:
        c, h = TestClient(admin_app), _h(admin_key)
        c.post("/admin/tenants", json={"tenant_id": "kampus-c"}, headers=h)
        mentah = c.post(
            "/admin/tenants/kampus-c/keys", json={}, headers=h,
        ).json()["api_key"]

        rincian = c.get("/admin/tenants/kampus-c", headers=h).json()

        rahasia = mentah.split(".", 1)[1]
        assert rahasia not in str(rincian)   # yang tersimpan hanya hash-nya
        assert rincian["keys"][0]["key_id"] in mentah   # key_id bukan rahasia

    def test_tenant_id_tidak_sah_ditolak(self, admin_app, admin_key) -> None:
        c, h = TestClient(admin_app), _h(admin_key)
        for buruk in ("Kampus A", "../etc", "a", "kampus/a"):
            r = c.post("/admin/tenants", json={"tenant_id": buruk}, headers=h)
            assert r.status_code in (400, 422), buruk

    def test_pembekuan_memutus_seluruh_kunci_tenant(
        self, admin_app, admin_key,
    ) -> None:
        # Diuji langsung di tingkat autentikasi, bukan lewat sebuah endpoint:
        # yang dipertanyakan adalah apakah KUNCINYA berhenti berlaku, bukan
        # apakah satu rute tertentu menolak.
        c, h = TestClient(admin_app), _h(admin_key)
        c.post("/admin/tenants", json={"tenant_id": "kampus-d"}, headers=h)
        kunci = c.post(
            "/admin/tenants/kampus-d/keys",
            json={"scopes": sorted(READER_SCOPES)}, headers=h,
        ).json()["api_key"]

        assert tenant_store.authenticate(kunci) is not None

        c.post(
            "/admin/tenants/kampus-d/status",
            json={"status": "suspended"}, headers=h,
        )
        # Tanpa mencabut kunci satu per satu, seluruh aksesnya berhenti.
        assert tenant_store.authenticate(kunci) is None

    def test_pencabutan_kunci(self, admin_app, admin_key) -> None:
        c, h = TestClient(admin_app), _h(admin_key)
        c.post("/admin/tenants", json={"tenant_id": "kampus-e"}, headers=h)
        terbit = c.post(
            "/admin/tenants/kampus-e/keys",
            json={"scopes": sorted(READER_SCOPES)}, headers=h,
        ).json()

        assert tenant_store.authenticate(terbit["api_key"]) is not None
        assert c.delete(
            f"/admin/tenants/kampus-e/keys/{terbit['key_id']}", headers=h,
        ).status_code == 204
        assert tenant_store.authenticate(terbit["api_key"]) is None

    def test_tenant_tidak_ada_menghasilkan_404(self, admin_app, admin_key) -> None:
        c = TestClient(admin_app)
        assert c.get(
            "/admin/tenants/entah-siapa", headers=_h(admin_key),
        ).status_code == 404
