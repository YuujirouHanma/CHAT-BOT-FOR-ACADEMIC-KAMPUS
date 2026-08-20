"""Upload route: save a document to storage and trigger indexing.

Unggahan adalah satu-satunya jalur di mana data dari luar berubah menjadi berkas
di disk milik kami, jadi pemeriksaannya berlapis:

- **Hak akses** — butuh `content:write`; kunci "aplikasi mahasiswa" tidak bisa
  menitipkan materi ke dalam index kampus.
- **Nama berkas** — dilucuti menjadi nama dasarnya saja, sehingga
  `../../etc/passwd` menyusut menjadi `passwd` dan tidak bisa menunjuk keluar.
- **Ekstensi** — hanya jenis yang memang dikenali pipeline.
- **Ukuran** — dibatasi sambil menulis, bukan setelahnya. Memeriksa
  `Content-Length` saja tidak cukup: nilainya dikirim klien dan bisa berbohong,
  sementara berkasnya sudah terlanjur memenuhi disk saat ketahuan.
- **Kuota simpan** — total pemakaian tenant diperiksa sebelum menerima berkas.

Berkas disimpan di `storage/{tenant_id}/{content_id}/`, terpisah secara fisik
antar pelanggan.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from src.api.auth import assert_body_tenant, client_ip, request_id, tenant_content_write
from src.api.dependencies import get_pipeline
from src.api.schemas import IndexResponse
from src.config import settings
from src.ingestion.validators import FileValidationError
from src.pipeline import RAGPipeline
from src.security import audit
from src.storage.content_store import (
    is_valid_content_id,
    storage_root,
    tenant_usage_bytes,
)
from src.tenancy import TenantContext
from src.utils.logger import logger

router = APIRouter(prefix="/documents", tags=["documents"])

_COPY_CHUNK = 1024 * 1024      # 1 MB


@router.post("/upload", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    content_id: str | None = Form(default=None),
    course_id: str | None = Form(default=None),
    course_name: str | None = Form(default=None),
    week: int | None = Form(default=None),
    tenant_id: str | None = Form(default=None),
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_content_write),
) -> IndexResponse:
    """Upload a document, save to storage/{tenant_id}/{content_id}/, and index.

    If content_id given → saved under storage/{tenant}/{content_id}/{filename}.
    Otherwise → saved to storage/{tenant}/_uploads/ with UUID prefix.

    course_id / course_name / week are optional — they power the guided catalog
    (mata kuliah → minggu → materi). If omitted they are derived from content_id
    (e.g. "sbd-minggu-2" → course "sbd", week 2).
    """
    rid = request_id(request)
    assert_body_tenant(tenant, tenant_id, request_id_=rid)
    if course_id:
        tenant.require_course(course_id)

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Nama berkas wajib ada",
        )
    if content_id is not None and not is_valid_content_id(content_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="content_id tidak valid",
        )

    nama = Path(file.filename).name
    ekstensi = nama.rsplit(".", 1)[-1].lower() if "." in nama else ""
    if ekstensi not in settings.all_known_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Jenis berkas '.{ekstensi}' tidak didukung",
        )

    batas = min(tenant.quota.max_upload_mb, settings.max_upload_size_mb) * 1024 * 1024
    terpakai = tenant_usage_bytes(tenant_id=tenant.tenant_id)
    if terpakai >= tenant.quota.max_storage_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Kuota penyimpanan tenant sudah penuh",
        )

    target_path = _save_file(
        file, nama=nama, content_id=content_id,
        tenant_id=tenant.tenant_id, max_bytes=batas,
    )

    try:
        result = await pipeline.index_document(
            file_path=target_path,
            content_id=content_id,
            course_id=course_id,
            course_name=course_name,
            week=week,
            tenant_id=tenant.tenant_id,
        )
    except FileValidationError as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc
    except Exception:
        target_path.unlink(missing_ok=True)
        logger.exception("Indexing failed for {}", nama)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal mengindeks berkas",
        ) from None

    audit.record(
        audit.CONTENT_UPLOAD, tenant_id=tenant.tenant_id, actor=tenant.key_id,
        request_id=rid, ip=client_ip(request),
        detail={
            "source_file": result.source_file,
            "content_id": result.content_id,
            "course_id": course_id,
            "chunks": result.chunks_created,
        },
    )
    return IndexResponse(
        source_file=result.source_file,
        elements_parsed=result.elements_parsed,
        chunks_created=result.chunks_created,
        points_stored=result.points_stored,
        content_id=result.content_id,
    )


def _save_file(
    file: UploadFile,
    *,
    nama: str,
    content_id: str | None,
    tenant_id: str,
    max_bytes: int,
) -> Path:
    """Tulis unggahan ke disk sambil menegakkan batas ukuran.

    Ditulis per potongan dan dihitung sambil jalan: begitu melewati batas,
    penulisan dihentikan dan berkas separuh jadi langsung dihapus. Membaca
    seluruh berkas ke memori lebih dulu — atau memercayai `Content-Length` —
    membuat satu permintaan besar cukup untuk menjatuhkan proses.
    """
    root = storage_root(tenant_id=tenant_id)
    dest_dir = root / content_id if content_id else root / "_uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (nama if content_id else f"{uuid.uuid4().hex}_{nama}")

    ditulis = 0
    try:
        with target.open("wb") as out:
            while True:
                potongan = file.file.read(_COPY_CHUNK)
                if not potongan:
                    break
                ditulis += len(potongan)
                if ditulis > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Ukuran berkas melampaui {max_bytes // (1024 * 1024)} MB",
                    )
                out.write(potongan)
    except Exception:
        target.unlink(missing_ok=True)
        raise

    logger.info("Saved upload to {} ({} bytes)", target, ditulis)
    return target
