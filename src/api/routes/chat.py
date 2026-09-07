"""Chat route: answer a user question, or guide the student to pick a material.

Satu endpoint, dua kemungkinan balasan:

- `mode="answer"`  → `answer` adalah jawaban atas pertanyaan mahasiswa.
- `mode="choices"` → `answer` adalah pertanyaan balik chatbot, dan `choices`
  berisi pilihan yang harus ditampilkan sebagai tombol.

Klien tidak perlu endpoint khusus untuk alur guided: mengklik sebuah tombol
cukup dikirim sebagai `question` biasa berisi label/nilai tombol itu, dan
resolver di `src/guided.py` yang mengenalinya. Jadi guided dan tanya-bebas
melewati kode yang sama.
"""
from __future__ import annotations

from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src import guided, learning_styles
from src.api.auth import assert_body_tenant, client_ip, request_id, tenant_chat
from src.api.dependencies import get_pipeline
from src.api.schemas import (
    Attachment,
    ChatContext,
    ChoiceInfo,
    FeedbackRequest,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)
from src.api.session import Session, session_store
from src.generation.notebook import build_notebook, safe_filename
from src.hitl.logger import log_feedback
from src.pipeline import RAGPipeline
from src.security import audit
from src.tenancy import TenantContext
from src.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["chat"])


def _context_of(session: Session) -> ChatContext:
    return ChatContext(
        course_id=session.course_id,
        course_name=session.course_name,
        weeks=list(session.weeks),
        week=session.weeks[0] if session.weeks else None,
        content_id=session.content_id,
        source_file=session.source_filter,
        style=session.style,
        topic=session.topic,
    )


def _with_back(choices: list[guided.Choice], step: str) -> list[guided.Choice]:
    """Tambahkan tombol kembali di AKHIR daftar pilihan.

    Di akhir, bukan di awal: pilihan yang sebenarnya harus lebih dulu terbaca.
    Tombol kembali adalah jalan keluar, bukan tawaran utama.
    """
    tombol = guided.back_choice(step)
    return choices if tombol is None else [*choices, tombol]


def _choices_reply(
    session: Session,
    message: str,
    step: str,
    choices: Sequence[guided.Choice] = (),
) -> QueryResponse:
    """Balasan bermode `choices` — chatbot bertanya, klien merender tombol.

    Pertanyaan template ikut disalin ke `recommendations` agar klien lama yang
    hanya membaca field itu tetap mendapat isinya.
    """
    info = [ChoiceInfo(label=c.label, value=c.value, kind=c.kind) for c in choices]
    # Langkah navigasi TIDAK masuk `history` (konteks LLM) tetapi TETAP masuk
    # transkrip, supaya percakapan dapat ditampilkan ulang persis seperti aslinya.
    session.record(
        "assistant", message, mode="choices", step=step,
        choices=[c.model_dump() for c in info],
    )
    session.persist()
    return QueryResponse(
        answer=message,
        sources=[],
        recommendations=[c.value for c in choices if c.kind == "question"],
        session_id=session.session_id,
        mode="choices",
        step=step,  # type: ignore[arg-type]
        choices=info,
        context=_context_of(session),
    )


async def _handle_quiz(
    body: QueryRequest, session: Session, pipeline: RAGPipeline, tenant_id: str,
) -> QueryResponse | None:
    """Jalankan kuis di dalam percakapan; None kalau pesan ini bukan urusan kuis.

    Kuis berlangsung lintas beberapa pesan, jadi soal dan jawaban disimpan di
    session. Soal disajikan satu per satu — bukan sekaligus — supaya mahasiswa
    yang belum terbiasa tidak kewalahan dan tetap terasa seperti percakapan.
    """
    # --- sedang mengerjakan kuis ---
    if session.quiz:
        if guided.wants_quiz_stop(body.question):
            session.stop_quiz()
            return _choices_reply(
                session,
                "Baik, kuisnya kita hentikan. Kamu bisa lanjut bertanya tentang "
                "materi ini kapan saja.",
                "question",
            )

        soal = session.current_quiz_question()
        if soal is not None:
            pilihan = guided.match_quiz_option(body.question, soal["options"])
            if pilihan is None:
                # Jangan diam-diam menghitungnya salah — tanyakan ulang.
                return _choices_reply(
                    session,
                    "Maaf, saya belum menangkap pilihanmu. Klik salah satu tombol "
                    "di bawah, atau ketik hurufnya saja (misalnya `B`).\n\n"
                    + guided.quiz_question_message(
                        session.quiz["index"], len(session.quiz["questions"]),
                        soal["question"],
                    ),
                    "quiz",
                    guided.quiz_option_choices(soal["options"]),
                )
            session.answer_quiz(pilihan)

        # masih ada soal berikutnya
        berikutnya = session.current_quiz_question()
        if berikutnya is not None:
            return _choices_reply(
                session,
                guided.quiz_question_message(
                    session.quiz["index"], len(session.quiz["questions"]),
                    berikutnya["question"],
                ),
                "quiz",
                guided.quiz_option_choices(berikutnya["options"]),
            )

        # --- selesai: nilai dan tampilkan pembahasan ---
        jawaban = list(session.quiz["answers"])
        session.stop_quiz()
        try:
            hasil = await pipeline.grade_quiz(
                session.content_id or "", session.source_filter or "", jawaban,
                session_id=session.session_id, tenant_id=tenant_id,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
        return _choices_reply(
            session,
            guided.quiz_result_message(
                hasil["score"], hasil["correct"], hasil["total"], hasil["results"],
            ),
            "question",
        )

    # --- diminta memulai kuis ---
    if not guided.wants_quiz(body.question):
        return None
    if not (session.content_id and session.source_filter):
        return None  # belum pilih materi → biarkan alur guided menuntun dulu

    soal = await pipeline.quiz(
        session.content_id, session.source_filter, model=body.model, tenant_id=tenant_id,
    )
    if not soal:
        return _choices_reply(
            session,
            "Maaf, saya belum berhasil menyusun kuis untuk materi ini. "
            "Kamu tetap bisa bertanya langsung tentang isinya.",
            "question",
        )
    session.start_quiz(soal)
    pertama = session.current_quiz_question() or soal[0]
    return _choices_reply(
        session,
        guided.quiz_start_message(session.source_filter, len(soal))
        + "\n\n---\n\n"
        + guided.quiz_question_message(0, len(soal), pertama["question"]),
        "quiz",
        guided.quiz_option_choices(pertama["options"]),
    )


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: Request,
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_chat),
) -> QueryResponse:
    rid = request_id(request)
    assert_body_tenant(tenant, body.tenant_id, request_id_=rid)
    # ABAC: kunci yang dibatasi pada sebagian mata kuliah tidak boleh memakai
    # kunci lain lewat parameter permintaan.
    if body.course_id:
        tenant.require_course(body.course_id)

    session = session_store.get_or_create(
        body.session_id, tenant_id=tenant.tenant_id, student_id=body.student_id,
    )
    session.record("user", body.question)

    # Konteks yang dikirim eksplisit oleh klien menang atas hasil resolusi teks.
    if any(v is not None for v in
           (body.content_id, body.source_filter, body.course_id, body.week, body.weeks)):
        session.set_guided(
            course_id=body.course_id,
            weeks=body.weeks or ([body.week] if body.week else None),
            content_id=body.content_id,
            source_filter=body.source_filter,
        )

    if body.style and learning_styles.get(body.style):
        session.style = body.style

    if body.guided:
        # Kuis mendahului navigasi: selama kuis berjalan, pesan mahasiswa adalah
        # jawaban soal — bukan pertanyaan maupun perintah pindah materi.
        quiz_reply = await _handle_quiz(body, session, pipeline, tenant.tenant_id)
        if quiz_reply is not None:
            return quiz_reply

        # Tombol kembali: mundur satu langkah SEBELUM menafsirkan pesan sebagai
        # pilihan baru. Tanpa didahulukan, label tombol ("Kembali — pilih materi
        # lain") ikut dibaca sebagai penyebutan materi dan justru maju, bukan
        # mundur. Hanya berlaku saat chatbot memang sedang menanyakan sesuatu;
        # di luar itu kata "kembali" bisa saja bagian dari pertanyaan biasa.
        # Kecocokan PERSIS pada tombol yang baru ditawarkan selalu menang atas
        # penebakan maksud. Label sokratik berbunyi "Saya balik bertanya…", dan
        # "balik" adalah salah satu frasa tombol kembali — tanpa aturan ini,
        # menekan tombol itu justru memundurkan alur, bukan memilih gayanya.
        if (
            session.awaiting
            and guided.wants_back(body.question)
            and not learning_styles.is_choice_label(body.question)
        ):
            # Klien boleh mengirim `value` tombol (nama langkah tujuan) alih-alih
            # labelnya; kalau begitu, nilainya itu sendiri yang menentukan tujuan.
            diminta = body.question.strip().lower()
            tujuan = (
                diminta if diminta in guided.BACK_TARGETS
                else guided.previous_step(session.awaiting)
            )
            if tujuan is not None:
                session.step_back_to(tujuan)
                turn = await pipeline.guided_turn(
                    "",                       # kosong: jangan tafsirkan ulang teksnya
                    course_id=session.course_id,
                    course_name=session.course_name,
                    weeks=list(session.weeks),
                    content_id=session.content_id,
                    source_file=session.source_filter,
                    style=session.style,
                    awaiting=session.awaiting,
                    model=body.model,
                    tenant_id=tenant.tenant_id,
                    allowed_courses=tenant.allowed_courses,
                )
                session.awaiting = turn.step
                return _choices_reply(
                    session, turn.message, turn.step,
                    _with_back(list(turn.choices), turn.step),
                )

        turn = await pipeline.guided_turn(
            body.question,
            course_id=session.course_id,
            course_name=session.course_name,
            weeks=list(session.weeks),
            content_id=session.content_id,
            source_file=session.source_filter,
            style=session.style,
            awaiting=session.awaiting,
            model=body.model,
            tenant_id=tenant.tenant_id,
            allowed_courses=tenant.allowed_courses,
        )
        # Hasil resolusi guided adalah sumber kebenaran konteks — dipakai apa
        # adanya, termasuk saat mengosongkan (mis. mahasiswa minta ganti materi).
        if turn.topic:
            session.topic = turn.topic
        session.style = turn.style
        session.apply_guided(
            course_id=turn.course_id,
            course_name=turn.course_name,
            weeks=turn.weeks,
            content_id=turn.content_id,
            source_filter=turn.source_file,
        )

        if not turn.answer_question:
            # Langkah navigasi TIDAK dimasukkan ke history: yang berguna untuk
            # pertanyaan lanjutan adalah alur belajar, bukan klik memilih menu.
            session.awaiting = turn.step
            choices = list(turn.choices)
            if turn.step == guided.STEP_QUESTION and session.content_id:
                # Kuis ditawarkan sebagai pilihan, bukan hanya lewat perintah —
                # mahasiswa sasaran belum tentu tahu fitur itu ada.
                # Label sengaja teks polos tanpa emoji: payload API dikonsumsi juga
                # oleh klien terminal/CLI yang konsolnya bukan UTF-8 (cp1252 di
                # Windows) dan akan gagal meng-encode-nya. Ikon urusan klien.
                choices.append(
                    guided.Choice(
                        label="Kerjakan kuis materi ini", value="kuis", kind="quiz",
                    )
                )
            return _choices_reply(
                session, turn.message, turn.step, _with_back(choices, turn.step),
            )

        session.awaiting = None

    logger.info(
        "Chat | tenant={} session={} course={} minggu={} content_id={} source={}",
        tenant.tenant_id, session.session_id[:8], session.course_id, session.weeks,
        session.content_id, session.source_filter,
    )
    audit.record(
        audit.CHAT_ASK, tenant_id=tenant.tenant_id, actor=tenant.key_id,
        request_id=rid, ip=client_ip(request),
        detail={"course_id": session.course_id, "content_id": session.content_id},
    )

    # ABAC ditegakkan pada cakupan pencarian yang SEBENARNYA, bukan hanya pada
    # `course_id` yang kebetulan dikirim di body. Permintaan yang tidak
    # menentukan mata kuliah menyapu SELURUH materi tenant — lebih luas daripada
    # jatah kunci yang dibatasi, jadi harus berhenti di sini. Memeriksa hanya
    # ketika klien mengirim `course_id` justru membuat pembatasan itu dapat
    # dilewati hanya dengan tidak mengirimkannya.
    tenant.require_course(session.course_id)

    try:
        result = await pipeline.query(
            question=body.question,
            content_id=session.content_id,
            source_filter=session.source_filter,
            session_id=session.session_id,
            model=body.model,
            level=body.level,
            # Riwayat SEBELUM pertanyaan ini, supaya pertanyaan lanjutan
            # menyambung alur dan tidak mengulang yang sudah dibahas.
            history=list(session.history),
            # Bila mahasiswa sudah memilih minggu tetapi belum satu materi,
            # pencarian tetap dibatasi ke minggu-minggu itu — bukan seluruh korpus.
            course_id=session.course_id if not session.content_id else None,
            weeks=list(session.weeks) if not session.content_id else None,
            style=session.style,
            tenant_id=tenant.tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        # Rincian galat tetap di log server. Membalikkannya ke klien akan
        # membocorkan nama pustaka, path berkas, dan kadang potongan kueri —
        # peta gratis bagi siapa pun yang sedang menjajaki sistem ini.
        logger.exception("Query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal memproses pertanyaan",
        ) from None

    session.add_turn("user", body.question)
    session.add_turn("assistant", result.answer)
    session.record(
        "assistant", result.answer, mode="answer", step="answer",
        sources=result.sources, interaction_id=result.interaction_id,
    )
    session.persist()

    # Gaya belajar "praktik" meminta LLM menulis kode; kode itu diubah menjadi
    # notebook secara deterministik dari jawaban yang sama, tanpa panggilan LLM
    # tambahan, sehingga isinya persis seperti yang dibaca mahasiswa.
    attachments: list[Attachment] = []
    spec = learning_styles.resolve(session.style)
    if spec.produces_notebook:
        isi = build_notebook(result.answer, title=body.question[:80])
        if isi:
            attachments.append(
                Attachment(
                    kind="notebook",
                    filename=safe_filename(session.source_filter or body.question[:40]),
                    content=isi,
                )
            )

    return QueryResponse(
        answer=result.answer,
        sources=[SourceInfo(**s) for s in result.sources],
        recommendations=result.recommendations,
        session_id=session.session_id,
        interaction_id=result.interaction_id,
        mode="answer",
        step="answer",
        attachments=attachments,
        context=_context_of(session),
    )


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    tenant: TenantContext = Depends(tenant_chat),
) -> None:
    assert_body_tenant(tenant, body.tenant_id, request_id_=request_id(request))
    log_feedback(
        interaction_id=body.interaction_id,
        rating=body.rating,
        issues=body.issues,
        comment=body.comment,
        tenant_id=tenant.tenant_id,
    )
