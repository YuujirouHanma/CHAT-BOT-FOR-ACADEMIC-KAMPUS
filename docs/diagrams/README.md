# Diagram RAGAcademic — Berkas Jadi & Cara Regenerasi

Folder ini berisi 10 diagram sistem dalam bentuk berkas gambar siap pakai, beserta
sumber dan skrip untuk membuatnya ulang.

```
docs/diagrams/
  src/                     10 berkas .mmd (sumber Mermaid, satu per gambar)
  svg/                     10 berkas .svg (VEKTOR — dipakai untuk naskah)
  png/                     10 berkas .png (raster 3x — pratinjau, slide, Word)
  extract.py               ekstrak ulang .mmd dari ARCHITECTURE_DIAGRAM.md
  render.ps1               render .mmd -> .svg + .png
  mermaid-config.json      gaya IEEE: hitam-putih, Times, garis tegas
  puppeteer-config.json    opsi peramban headless
```

Sumber kebenaran tetap [../ARCHITECTURE_DIAGRAM.md](../ARCHITECTURE_DIAGRAM.md).
Ubah diagram di sana, lalu jalankan ulang kedua skrip di bawah.

## Daftar gambar

Semua gambar sudah diperiksa agar muat pada kolom cetak IEEE. Kolom terakhir adalah
lebar pasang yang disarankan.

| Berkas | Judul | Pasang di |
|---|---|---|
| **`fig00-ikhtisar-alur-sistem-dalam-satu-gambar`** | **Ikhtisar seluruh sistem dalam satu gambar** | **1 kolom (4,1 in)** |
| `fig01-arsitektur-sistem-berlapis` | Arsitektur Sistem Berlapis | 1 kolom |
| `fig02-alur-pipa-indexing-luring-a` | Indexing (a): berkas → ParsedElement | 1 kolom |
| `fig02-alur-pipa-indexing-luring-b` | Indexing (b): pengayaan → penyimpanan | 1 kolom |
| `fig03-alur-pipa-kueri-daring-lima-tahap` | Alur Pipa Kueri Daring Lima-Tahap | 1 kolom |
| `fig04-mekanisme-hybrid-retrieval-dengan-fusi-rrf` | Hybrid Retrieval dengan Fusi RRF | 2 kolom |
| `fig05-diagram-sekuens-unggah-indexing-materi` | Sekuens: Unggah & Indexing | 1 kolom |
| `fig06-diagram-sekuens-tanya-jawab-multimodal` | Sekuens: Tanya-Jawab Multimodal | 1 kolom |
| `fig07-mesin-keadaan-navigasi-katalog-terpandu` | Mesin Keadaan Katalog Terpandu | 1 kolom |
| `fig08-diagram-deployment` | Diagram Deployment | 1 kolom |
| `fig09-skema-data-metadata-payload` | Skema Data & Metadata Payload | 1 kolom |
| `fig10-lingkar-umpan-balik-human-in-the-loop` | Lingkar Umpan Balik HITL | 1 kolom |

Fig. 2 sengaja dipecah menjadi dua panel: sebagai satu gambar utuh, pipa indexing
setinggi 9,4 inci dan tidak muat di kolom cetak mana pun. Pasang keduanya sebagai
Fig. 2(a) dan Fig. 2(b) dalam satu lingkungan `figure`.

**Kalau naskah hanya boleh memuat sedikit gambar, pakai `fig00` saja.** Gambar itu
berdiri sendiri: satu gambar yang memuat alur penuh dari materi kuliah masuk sampai
jawaban keluar, tanpa perlu gambar pendukung. Di naskah, jadikan ia Fig. 1 lalu
geser penomoran gambar berikutnya.

## Regenerasi

```powershell
python docs\diagrams\extract.py
powershell -ExecutionPolicy Bypass -File docs\diagrams\render.ps1
```

`render.ps1` mengambil `@mermaid-js/mermaid-cli` lewat `npx` (butuh Node.js dan
koneksi internet pada pemakaian pertama).

Perenderan berjalan di peramban headless. Skrip otomatis memakai **Chrome atau Edge
yang sudah terpasang di sistem**, sehingga tidak perlu mengunduh Chromium terpisah
sebesar ~150 MB. Bila keduanya tidak ada, pasang manual:

```powershell
npx puppeteer browsers install chrome
```

Untuk menunjuk peramban tertentu, set variabel lingkungan sebelum menjalankan skrip:

```powershell
$env:PUPPETEER_EXECUTABLE_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

## Memakai gambar di naskah

### Word / LibreOffice

Sisipkan berkas dari `svg/`. Word 2016 ke atas menerima SVG dan tetap tajam saat
dicetak maupun diekspor ke PDF. Jangan pakai PNG untuk naskah final — penerbit
menolak raster di bawah 600 dpi untuk *line art*.

### LaTeX

IEEE menghendaki PDF/EPS. Konversi SVG sekali saja:

```powershell
# butuh Inkscape terpasang
Get-ChildItem docs\diagrams\svg\*.svg | ForEach-Object {
    $out = $_.FullName -replace '\.svg$', '.pdf'
    & inkscape $_.FullName --export-type=pdf --export-text-to-path --export-filename=$out
}
```

`--export-text-to-path` mengubah teks menjadi kurva sehingga fonta tidak bergeser
di komputer penerbit. Lalu di naskah:

```latex
\begin{figure}[!t]
  \centering
  \includegraphics[width=\columnwidth]{fig01-arsitektur-sistem-berlapis.pdf}
  \caption{Arsitektur berlapis RAGAcademic. Panah padat menyatakan aliran
  permintaan/data internal; panah putus-putus menyatakan pemanggilan layanan
  eksternal melalui jaringan.}
  \label{fig:arsitektur}
\end{figure}
```

Untuk gambar lebar (Fig. 2, 5, 6, 9) pakai lingkungan dua kolom:

```latex
\begin{figure*}[!t]
  \centering
  \includegraphics[width=\textwidth]{fig02-alur-pipa-indexing-luring.pdf}
  \caption{...}
\end{figure*}
```

### Menyunting sebelum dipakai

SVG di sini di-render dengan `htmlLabels: false`, sehingga seluruh teks berupa
elemen `<text>` SVG asli — bukan `<foreignObject>`. Artinya berkas bisa dibuka di
Inkscape atau Illustrator dan setiap label, kotak, serta panah dapat digeser satu
per satu tanpa merusak tata letak.

## Batas ukuran IEEE

| Jenis | Lebar maksimum |
|---|---|
| Satu kolom | 3,5 inci (88 mm) |
| Dua kolom | 7,16 inci (181 mm) |

Diagram bertingkat seperti Fig. 2 dan Fig. 5 hampir pasti butuh lebar dua kolom,
atau dipecah menjadi dua panel. Lihat bagian II.E pada
[../ARCHITECTURE_DIAGRAM.md](../ARCHITECTURE_DIAGRAM.md) untuk rekomendasi gambar
mana yang masuk naskah dan mana yang sebaiknya ke lampiran.
