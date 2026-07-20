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

COPY src ./src

# Model AI (BGE-M3 ~1GB, reranker ~600MB) diunduh saat pertama run, bukan saat build.
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
