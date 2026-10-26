"""Registry of selectable generation models across providers.

The UI lets users switch model per request. Each request may send a `model`
key from this registry; the backend resolves it to (provider, model id) and
builds the right OpenAI-compatible client on the fly.

`available()` returns only models whose provider has an API key configured,
so the UI dropdown never offers a model that would fail.

To add/adjust models, edit `_SPECS` below — that is the single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import settings


@dataclass(frozen=True)
class ModelSpec:
    key: str          # stable id the UI/request uses, e.g. "qwen3.7-plus"
    label: str        # human display name
    provider: str     # openrouter | openai | groq | gemini | ...
    model: str        # provider-specific model id
    vision: bool      # supports image input


_SPECS: list[ModelSpec] = [
    ModelSpec("qwen3.7-flash", "Qwen 3.7 Flash", "openrouter", "qwen/qwen3.7-flash", True),
    ModelSpec("qwen3.7-plus", "Qwen 3.7 Plus", "openrouter", "qwen/qwen3.7-plus", True),
    ModelSpec("qwen3.6-flash", "Qwen 3.6 Flash", "openrouter", "qwen/qwen3.6-flash", True),
    # CATATAN, supaya tidak ditambahkan kembali: tabel `generation_models` di
    # portal (api/schema.php) memuat baris ketiga "qwen/qwen2.5-72b-instruct".
    # Id itu DITOLAK penyedianya — diuji 26 Oktober 2026, OpenRouter membalas
    # 400 "qwen/qwen2.5-72b-instruct is not a valid model ID". Jangan daftarkan
    # di sini sampai ada id yang benar-benar dilayani.
    ModelSpec("gpt-4o-mini", "GPT-4o mini", "openai", "gpt-4o-mini", True),
    ModelSpec("gemini-flash", "Gemini 2.5 Flash", "gemini", "gemini-2.5-flash", True),
    ModelSpec("gemini-pro", "Gemini 2.5 Pro", "gemini", "gemini-2.5-pro", True),
    ModelSpec("groq-llama-70b", "Llama 3.3 70B (Groq)", "groq", "llama-3.3-70b-versatile", False),
]

_REGISTRY: dict[str, ModelSpec] = {s.key: s for s in _SPECS}


def get(key: str | None) -> ModelSpec | None:
    """Resolve a registry key to its spec, or None if unknown/empty."""
    if not key:
        return None
    return _REGISTRY.get(key)


def kunci_untuk(provider: str | None, model_id: str | None) -> str | None:
    """Kunci registry untuk sepasang (penyedia, id model) dari kosakata LMS.

    MENGAPA FUNGSI INI ADA
    Portal kampus menyebut model dengan dua medan — `model_provider` dan
    `model_name` — dan `model_name` berisi id milik PENYEDIA (mis.
    "qwen/qwen3.7-plus"), karena itu yang tersimpan di tabel
    `generation_models`. Registry ini memakai kunci pendeknya sendiri
    ("qwen3.7-plus"), dengan penyedia yang sudah melekat pada entrinya.

    Tanpa penerjemahan ini, id penyedia diteruskan apa adanya sebagai kunci,
    `get()` mengembalikan None, dan permintaan dilayani model BAWAAN —
    sementara pemanggil yakin modelnya sudah diganti. Terukur: tiga panggilan
    dengan `model_name=qwen/qwen3.7-plus` seluruhnya tercatat di buku pemakaian
    sebagai qwen3.7-flash, dan satu-satunya tanda hanya peringatan di log
    server. Untuk fitur yang gunanya MEMBANDINGKAN model dan biayanya, jawaban
    yang benar dari model yang salah lebih buruk daripada galat.

    Penyedia dicocokkan bila diberikan. Bila tidak, id model saja dipakai dan
    hanya diterima ketika ia menunjuk satu entri — dua entri dengan id sama di
    penyedia berbeda adalah keadaan yang tidak boleh ditebak.

    Mengembalikan None bila tidak ada yang cocok; pemanggil meneruskan nilai
    aslinya supaya peringatan "Unknown model" tetap muncul apa adanya.
    """
    nama = (model_id or "").strip()
    if not nama:
        return None

    # Sudah berupa kunci registry: diterima, supaya pemanggil yang mengirim
    # kunci di medan `model_name` tidak ikut tergelincir.
    if nama in _REGISTRY:
        return nama

    pv = (provider or "").strip().lower()
    cocok = [s for s in _SPECS if s.model == nama and (not pv or s.provider == pv)]
    if len(cocok) == 1:
        return cocok[0].key
    return None


def all_specs() -> list[ModelSpec]:
    return list(_SPECS)


def available() -> list[ModelSpec]:
    """Only models whose provider has an API key configured."""
    return [s for s in _SPECS if settings.api_key_for(s.provider)]
