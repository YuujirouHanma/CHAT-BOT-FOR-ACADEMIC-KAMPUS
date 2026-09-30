"""Course catalog: derive the mata-kuliah → minggu → materi hierarchy.

The guided UI needs to list courses, weeks, and materials so students can
navigate by clicking instead of typing. Rather than require a separate LMS,
we derive this hierarchy from what is actually indexed — so it auto-updates
when new material is uploaded, and never shows a week/material with no content.

Each material carries course_id / course_name / week. These may be provided
explicitly at upload time; when absent we parse them from content_id, which by
convention looks like "<course>-minggu-<n>" (also accepts week/w and _ or /
separators, e.g. "sbd_minggu_2", "sbd/minggu_2", "algo-w3").
"""
from __future__ import annotations

import re

# content_id → (course slug, week number). Requires a separator before the
# week keyword so plain ids like "class-3" are treated as a course, not week 3.
_WEEK_RE = re.compile(
    r"^(?P<course>.+?)[-_/](?:minggu|week|w)[-_]?(?P<week>\d{1,2})$",
    re.IGNORECASE,
)


def parse_content_id(content_id: str | None) -> tuple[str | None, int | None]:
    """Parse (course_id, week) from a content_id.

    Returns (None, None) for an empty id. When no week pattern is present the
    whole id is taken as the course_id and week is None.
    """
    if not content_id:
        return None, None
    cid = content_id.strip()
    match = _WEEK_RE.match(cid)
    if not match:
        return (cid or None), None
    course = match.group("course").strip().strip("-_/")
    return (course or None), int(match.group("week"))


def humanize_course(course_id: str | None) -> str | None:
    """Nama tampilan hasil TEBAKAN MEKANIS dari slug.

    'dasar-pemrograman' → 'Dasar Pemrograman'; slug pendek beralfabet
    (mis. 'sbd') dianggap akronim dan dikapitalkan.

    Tebakan ini netral terhadap kampus mana pun karena hanya mengolah slug yang
    dikirim pengunggah sendiri — ia tidak pernah memasangkan arti milik satu
    kampus ke kode mata kuliah kampus lain. Nama yang benar-benar dikenali
    mahasiswa datang dari data tenant: parameter `course_name` saat unggah,
    yang disimpan `QdrantStore` di payload.

    Hasilnya harus tetap murni mekanis karena `QdrantStore.list_courses`
    memakai fungsi ini sebagai pembanding untuk mengenali "nama ini cuma
    tebakan, jadi nama eksplisit dari pengunggah boleh menggantikannya".
    """
    if not course_id:
        return None
    words = [w for w in re.split(r"[-_/\s]+", course_id.strip()) if w]
    if len(words) == 1 and words[0].isalpha() and len(words[0]) <= 4:
        return words[0].upper()
    return " ".join(w.capitalize() for w in words) or None


def display_name(course_id: str | None) -> str | None:
    """Nama yang ditampilkan kepada mahasiswa — pembungkus tipis `humanize_course`.

    Dulu fungsi ini terpisah karena ada kamus nama kurasi di kode yang
    didahulukan sebelum tebakan mekanis. Kamus itu dibuang: isinya nama mata
    kuliah milik SATU kampus, padahal satu pemasangan melayani banyak kampus.
    Kampus lain yang kebetulan memakai slug sama akan melihat nama mata kuliah
    kampus lain pada tombol pilihannya — dan nama itu bukan sekadar tampilan:
    ia ikut ke prompt LLM lalu muncul di jawaban chatbot. Karena itu nama
    kurasi di kode tidak boleh kembali; satu-satunya sumber nama khusus adalah
    data tenant lewat `course_name` saat unggah.

    Namanya dipertahankan agar pemanggil (mis. `src/pipeline.py`) tetap menandai
    "ini titik penyajian", terpisah dari pemakaian `humanize_course` sebagai
    pembanding di dalam logika katalog.
    """
    return humanize_course(course_id)


def resolve_course_week(
    content_id: str | None,
    course_id: str | None = None,
    course_name: str | None = None,
    week: int | None = None,
) -> tuple[str | None, str | None, int | None]:
    """Combine explicit values (which win) with values parsed from content_id."""
    parsed_course, parsed_week = parse_content_id(content_id)
    course_id = course_id or parsed_course
    week = week if week is not None else parsed_week
    course_name = course_name or display_name(course_id)
    return course_id, course_name, week
