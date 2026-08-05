# Reference image for the RAGAcademic backend.
# Tim BE boleh pakai langsung atau menyamakan Dockerfile mereka — yang WAJIB
# adalah: (1) paket sistem OCR di bawah, (2) env QDRANT_MODE=server (di compose).
FROM python:3.12-slim

# System deps:
# - tesseract-ocr + tesseract-ocr-ind : OCR untuk PDF hasil scan (bahasa Indonesia)
# - poppler-utils                     : render PDF -> image untuk OCR (pdf2image)
# - libmagic1                         : deteksi tipe file (unstructured)
# - libgl1, libglib2.0-0              : dependensi image/opencv (unstructured)
# - ffmpeg                            : ekstrak audio dari video/audio untuk transkripsi
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-ind \
        poppler-utils \
        libmagic1 \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies dulu (layer cache) lalu salin kode.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Data NLTK untuk unstructured — WAJIB diunduh saat BUILD.
# Kalau tidak ada, unstructured mengunduhnya saat RUNTIME dari
# https://utic-public-cf.s3.amazonaws.com/nltk_data_3.8.2.tar.gz yang sering
# membalas 403 Forbidden di dalam container → parsing pptx/docx GAGAL.
# Di sini kita ambil dari server resmi NLTK dan simpan ke lokasi yang dicari
# unstructured (check_for_nltk_package menelusuri path yang berakhiran nltk_data).
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -m nltk.downloader -d ${NLTK_DATA} punkt_tab averaged_perceptron_tagger_eng \
    && python -c "from unstructured.nlp.tokenize import check_for_nltk_package as c; \
assert c(package_name='punkt_tab', package_category='tokenizers'), 'punkt_tab tidak ditemukan'; \
assert c(package_name='averaged_perceptron_tagger_eng', package_category='taggers'), 'tagger tidak ditemukan'; \
print('NLTK data siap — unstructured tidak akan mengunduh saat runtime')"

COPY src ./src

# Model AI (BGE-M3 ~1GB, reranker ~600MB) diunduh saat pertama run, bukan saat build.
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
