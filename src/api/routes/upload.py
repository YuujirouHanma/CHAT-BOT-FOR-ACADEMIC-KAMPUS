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
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)

from src.api.auth import (
    assert_body_tenant,
    client_ip,
    request_id,
    tenant_content_read,
    tenant_content_write,
)
from src.api.dependencies import get_pipeline
from src.api.jobs import UploadJob, job_store
from src.api.schemas import (
    IndexedFileInfo,
    IndexedFilesResponse,
    IndexResponse,
    UploadJobResponse,
    UploadJobStatusResponse,
)
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

# Satu kalimat untuk semua kegagalan pengindeksan: sebabnya masuk log,
# bukan ke pemanggil. Menyebut sebab teknis ke portal kampus tidak
# menolong penggunanya dan membocorkan bentuk dalam sistem.
_INDEX_FAILED = "Gagal mengindeks berkas"

router = APIRouter(prefix="/documents", tags=["documents"])

_COPY_CHUNK = 1024 * 1024      # 1 MB


def _minggu(week: int | None, section_id: int | None) -> int | None:
    """Satu nomor minggu dari dua nama medan yang dipakai pemanggil berbeda.

    Portal kampus memakai kosakata LMS (`section_id`); payload indeks dan
    seluruh jalur pencarian memakai `week`. Keduanya diterima di SETIAP endpoint
    unggah, bukan satu nama di satu endpoint dan nama lain di endpoint
    sebelahnya: perbedaan itu membuat portal kehilangan minggunya tanpa galat
    apa pun, dan gejalanya baru muncul jauh kemudian sebagai "penyaring minggu
    tidak berpengaruh".

    `week` menang bila keduanya terkirim — ia nama yang sebenarnya dipakai
    payload, jadi pemanggil yang menyebutnya sedang berbicara lebih spesifik.
    """
    return week if week is not None else section_id


def _periksa_unggahan(
    file: UploadFile,
    *,
    content_id: str | None,
    course_id: str | None,
    tenant: TenantContext,
) -> tuple[str, int]:
    """Pemeriksaan yang WAJIB dilalui setiap unggahan, apa pun jalurnya.

    Mengembalikan (nama berkas yang sudah dibersihkan, batas ukuran dalam bita).
    Ditulis satu kali di sini, bukan diulang di tiap endpoint: ketika jalur
    latar belakang ditambahkan dengan pemeriksaannya sendiri, ia melewatkan
    seluruhnya dan tidak ada yang menyadarinya sampai diperiksa baris demi
    baris.
    """
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
    return nama, batas


@router.post("/upload", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    content_id: str | None = Form(default=None),
    course_id: str | None = Form(default=None),
    course_name: str | None = Form(default=None),
    week: int | None = Form(default=None),
    section_id: int | None = Form(default=None),
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

    `section_id` diterima sebagai nama lain `week`: portal kampus memakai
    kosakata LMS. Sebelumnya endpoint ini hanya mengenal `week` sementara jalur
    latar belakang hanya mengenal `section_id`, sehingga portal yang mengirim
    `section_id` ke sini kehilangan minggunya TANPA satu pun galat — berkasnya
    terindeks, tetapi tidak pernah muncul ketika mahasiswa mempersempit minggu
    aktif, dan evaluasi atas rentang minggu tidak menemukan materi apa pun.
    """
    week = _minggu(week, section_id)
    rid = request_id(request)
    assert_body_tenant(tenant, tenant_id, request_id_=rid)
    if course_id:
        tenant.require_course(course_id)

    nama, batas = _periksa_unggahan(
        file, content_id=content_id, course_id=course_id, tenant=tenant,
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

# --- Unggah latar belakang dan pengelolaan berkas ---------------------------
# Dipindahkan dari layanan tim back-end. Unggahan serentak menahan koneksi
# HTTP selama pengindeksan berlangsung — buku 11 MB terukur sekitar lima
# menit — sehingga peramban kerap menyerah lebih dulu. Jalur latar belakang
# mengembalikan id pekerjaan yang ditanyakan berkala.


@router.post(
    "/upload/background", response_model=UploadJobResponse, status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document_background(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    content_id: str | None = Form(default=None),
    section_id: int | None = Form(default=None),
    week: int | None = Form(default=None),
    course_id: str | None = Form(default=None),
    course_name: str | None = Form(default=None),
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_content_write),
) -> UploadJobResponse:
    """Sama seperti /upload, tapi tidak menunggu indexing selesai.

    Menyimpan berkas ke disk masih dilakukan di sini (cepat — bukan bagian
    yang lambat), lalu proses parse + ringkas LLM + chunk + embed berjalan di
    latar belakang lewat BackgroundTasks (di pekerja proses yang sama, setelah
    respons ini terkirim). Status jobnya dicek lewat
    GET /documents/upload/background/{job_id} — lihat src/api/jobs.py untuk
    alasan kenapa statusnya disimpan di job_store (bukan variabel lokal):
    permintaan cek status berikutnya bisa mendarat di pekerja uvicorn LAIN.

    Dibuat sebagai endpoint terpisah, bukan mengubah /upload: endpoint lama
    dipakai apa adanya oleh "Indeks ulang semua" di frontend (proses batch
    yang sudah menunggu satu per satu), jadi mengubah bentuk responsnya di
    tempat akan merembet ke situ tanpa perlu.
    """
    content_id = content_id or None  # string kosong dari klien lama = tidak diisi
    minggu = _minggu(week, section_id)
    # Pemeriksaan yang sama dengan jalur serentak. Tanpa ini, memilih endpoint
    # latar belakang berarti melewati pembatasan mata kuliah dan kuota kampus.
    nama, batas = _periksa_unggahan(
        file, content_id=content_id, course_id=course_id, tenant=tenant,
    )
    target_path = _save_file(
        file, nama=nama, content_id=content_id,
        tenant_id=tenant.tenant_id, max_bytes=batas,
    )
    job = await job_store.create(tenant_id=tenant.tenant_id)
    background_tasks.add_task(
        _run_index_job, job.job_id, pipeline, target_path, content_id, minggu,
        tenant.tenant_id, course_id, course_name,
    )
    return UploadJobResponse(job_id=job.job_id, status=job.status)


@router.get("/upload/background/{job_id}", response_model=UploadJobStatusResponse)
async def get_upload_job_status(
    job_id: str,
    tenant: TenantContext = Depends(tenant_content_write),
) -> UploadJobStatusResponse:
    """Cek status job dari POST /documents/upload/background.

    Job kampus lain dibalas 404 sama seperti job yang tidak pernah ada:
    membedakan keduanya memberi tahu bahwa sebuah job_id memang ada.
    """
    job = await job_store.get(job_id, tenant_id=tenant.tenant_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job tidak ditemukan")

    return UploadJobStatusResponse(
        job_id=job.job_id, status=job.status, result=job.result, error=job.error,
    )


async def _run_index_job(
    job_id: str,
    pipeline: RAGPipeline,
    target_path: Path,
    content_id: str | None,
    week: int | None,
    tenant_id: str,
    course_id: str | None = None,
    course_name: str | None = None,
) -> None:
    """Badan try/except-nya sama persis dengan upload_document() di atas —
    bedanya hasil/galat ditulis ke job_store, bukan dikembalikan lewat HTTP,
    karena responsnya sudah terkirim duluan saat fungsi ini mulai jalan."""
    job = await job_store.get(job_id, tenant_id=tenant_id)
    if job is None:
        job = UploadJob(job_id=job_id, tenant_id=tenant_id)
    job.status = "running"
    await job_store.save(job)

    try:
        result = await pipeline.index_document(
            file_path=target_path, content_id=content_id, week=week,
            course_id=course_id, course_name=course_name,
            tenant_id=tenant_id,
        )
    except FileValidationError as exc:
        target_path.unlink(missing_ok=True)
        job.status = "error"
        job.error = str(exc)
        await job_store.save(job)
        return
    except Exception:
        target_path.unlink(missing_ok=True)
        logger.exception(
            "Indexing latar belakang gagal untuk {} (tenant={})", target_path.name, tenant_id,
        )
        job.status = "error"
        job.error = _INDEX_FAILED
        await job_store.save(job)
        return

    job.status = "done"
    job.result = {
        "source_file": result.source_file,
        "elements_parsed": result.elements_parsed,
        "chunks_created": result.chunks_created,
        "points_stored": result.points_stored,
        "content_id": result.content_id,
    }
    await job_store.save(job)


@router.delete("/content/{content_id}", status_code=status.HTTP_200_OK)
async def purge_content(
    content_id: str,
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_content_write),
) -> dict:
    """Hapus seluruh chunk terindeks milik satu content_id di kampus pemanggil.

    Pasangan dari indeks ulang menyeluruh di frontend: chunk lama tersimpan
    dengan nama berkas asli sementara basis data hanya menyimpan nama tersimpan
    yang sudah disanitasi, jadi menghapus per berkas selalu menyisakan duplikat.

    Berkas fisik di storage/ TIDAK ikut terhapus — hanya isi indeks.
    """
    if not content_id.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="content_id required")

    try:
        await pipeline._store.delete_by_content_id(content_id, tenant_id=tenant.tenant_id)
    except Exception:
        logger.exception("Purge failed for content_id={} (tenant={})", content_id, tenant.tenant_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Gagal menghapus indeks") from None

    return {"content_id": content_id, "purged": True}


@router.delete("/file", status_code=status.HTTP_200_OK)
async def delete_file_index(
    content_id: str = Query(..., min_length=1),
    source_file: str = Query(..., min_length=1),
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_content_write),
) -> dict:
    """Hapus chunk milik SATU berkas dari indeks.

    Pasangan dari penghapusan berkas di portal dosen. Tanpa ini, menghapus
    berkas hanya melenyapkannya dari disk dan basis data sementara chunk-nya
    tetap hidup di Qdrant — sehingga materi yang sudah dihapus terus muncul di
    jawaban mahasiswa, lengkap dengan kutipan sumbernya.

    ``source_file`` harus nama berkas ASLI saat diunggah (kolom ``original_name``
    di basis data), bukan nama tersimpan yang sudah disanitasi. Keduanya berbeda,
    dan nama tersimpan tidak akan cocok dengan apa pun di payload Qdrant.

    Berkas fisik di storage/ TIDAK ikut terhapus — hanya isi indeks.
    """
    try:
        sebelum = await pipeline._store.scroll_all(
            content_id=content_id,
            source_filter=source_file,
            payload_fields=["source_file"],
            tenant_id=tenant.tenant_id,
        )
        await pipeline._store.delete_by_source(
            source_file, content_id=content_id, tenant_id=tenant.tenant_id,
        )
    except Exception:
        logger.exception(
            "Gagal menghapus indeks berkas {} (content_id={}, tenant={})",
            source_file, content_id, tenant.tenant_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal menghapus indeks berkas",
        ) from None

    # chunks_deleted = 0 bukan galat: berkas yang tidak menyisakan chunk saat
    # diindeks (mis. slide gambar tanpa teks) memang tidak punya apa pun untuk
    # dihapus. Angkanya dikembalikan supaya pemanggil bisa membedakan "tidak ada
    # apa-apa di sana" dari "nama berkasnya salah".
    return {
        "content_id": content_id,
        "source_file": source_file,
        "chunks_deleted": len(sebelum),
    }


@router.get("/indexed", response_model=IndexedFilesResponse)
async def list_indexed(
    content_id: str | None = Query(default=None),
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_content_read),
) -> IndexedFilesResponse:
    """Daftar berkas yang benar-benar ADA di indeks kampus ini, per minggu.

    Dipakai untuk mengaudit indeks terhadap basis data: apa pun di sini yang
    tidak punya pasangan di ``section_content`` adalah sisa berkas yang sudah
    dihapus, dan masih ikut terambil saat mahasiswa bertanya.
    """
    chunks = await pipeline._store.scroll_all(
        content_id=content_id,
        payload_fields=["source_file", "section_id", "content_id"],
        tenant_id=tenant.tenant_id,
    )

    hitung: dict[tuple[str | None, int | None, str], int] = {}
    for item in chunks:
        p = item["payload"]
        kunci = (p.get("content_id"), p.get("section_id"), p.get("source_file") or "")
        hitung[kunci] = hitung.get(kunci, 0) + 1

    berkas = [
        IndexedFileInfo(
            content_id=cid,
            section_id=sid,
            source_file=sf,
            chunks=jumlah,
        )
        for (cid, sid, sf), jumlah in sorted(
            hitung.items(), key=lambda x: (str(x[0][0]), x[0][1] or 0, x[0][2])
        )
    ]

    return IndexedFilesResponse(
        files=berkas,
        total_files=len(berkas),
        total_chunks=sum(b.chunks for b in berkas),
    )
