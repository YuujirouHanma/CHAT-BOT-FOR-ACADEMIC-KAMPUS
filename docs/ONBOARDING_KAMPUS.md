# Runbook: Menambahkan Kampus Baru

**Untuk:** siapa pun yang menyiapkan satu kampus (tenant) baru di pemasangan
RAGAcademic yang sudah berjalan.
**Hasil akhir:** kampus itu dapat mengunggah materinya sendiri, mahasiswanya
dapat bertanya, dan sudah **dibuktikan** tidak dapat melihat data kampus lain.

Runbook ini hanya berisi langkah operasional. Alasan di balik rancangan
keamanannya — kenapa `tenant_id` hanya lahir dari kunci, kenapa hash kunci
memakai HMAC, kenapa berkas dipisah secara fisik — ada di
[`docs/SECURITY.md`](SECURITY.md) dan tidak diulang di sini.

Perintah ditulis untuk Windows PowerShell dari akar proyek. Pada PowerShell,
`curl` adalah alias `Invoke-WebRequest`, jadi tulis **`curl.exe`** bila ingin
perilaku curl sungguhan. Pada Linux/macOS ganti `.venv/Scripts/python.exe`
menjadi `.venv/bin/python`.

---

## 0. Peta singkat

| Tahap | Alat | Bagian |
|---|---|---|
| Siapkan pemasangan (sekali saja) | `.env` | [1](#1-prasyarat-sebelum-kampus-pertama) |
| Buat tenant + kunci | `scripts/tenantctl.py` | [2](#2-membuat-tenant-dan-menerbitkan-kuncinya) |
| Masukkan materi | HTTP `POST /documents/upload` | [3](#3-mengunggah-materi) |
| Kalender akademik kampus | `scripts/tenantctl.py` | [4](#4-kalender-akademik-kampus) |
| **Verifikasi isolasi** | HTTP + `verify-audit` | [5](#5-verifikasi-isolasi--jangan-dilewati) |
| Akhiri langganan | `tenantctl` + `/admin` | [6](#6-mencabut-membekukan-menghapus) |

Dua jalur pengelolaan tersedia dan keduanya setara:

- **CLI `scripts/tenantctl.py`** — jalur yang dianjurkan. Tidak menambah
  permukaan serang apa pun karena berjalan di mesin server.
- **HTTP `/admin/*`** — hanya hidup bila `ENABLE_ADMIN_API=true`. Berguna kalau
  penyediaan tenant dikerjakan sistem lain. Perhatikan: endpoint ini dapat
  menerbitkan kunci untuk tenant **mana pun**, jadi jalankan pada instance
  terpisah yang tidak terjangkau dari internet.

---

## 1. Prasyarat sebelum kampus pertama

Kerjakan sekali per pemasangan, bukan per kampus.

### 1.1 Setelan yang WAJIB benar di produksi

Di produksi (`APP_ENV=production`) empat butir pertama ini **menggagalkan
startup** — server menolak jalan, bukan sekadar mencatat peringatan. Itu
disengaja: setiap butirnya berarti data satu kampus dapat terbaca kampus lain.
Pemeriksaannya ada di `_check_security_config()` (`src/api/main.py`).

| Setelan | Nilai yang benar | Akibat bila salah |
|---|---|---|
| `TENANT_KEY_PEPPER` | string acak 32 byte | **Startup gagal.** Hash kunci API, MAC jejak audit, dan indeks buta PII berjalan tanpa rahasia sisi server — salinan `data/tenants/` yang bocor cukup untuk memalsukan kunci. |
| (jumlah tenant) | minimal 1 tenant terdaftar | **Startup gagal.** Tanpa tenant, permintaan tanpa kunci akan jatuh ke tenant pengembangan yang memegang hampir seluruh hak. |
| `RAGACADEMIC_API_KEY` | **kosong** | **Startup gagal.** Kunci global warisan tidak membawa identitas tenant, jadi isolasi tidak dapat ditegakkan untuk permintaan yang memakainya. |
| `CORS_ALLOW_ORIGINS` | daftar asal sungguhan | **Startup gagal.** `*` bersama kredensial membuat halaman web mana pun dapat memanggil API ini memakai kunci korban. |
| `TRUSTED_HOSTS` | nama host layanan | Peringatan. `*` membuat permintaan dengan header `Host` palsu tetap dilayani. |
| `ENCRYPT_PII` + `PII_KEK` | `true`, KEK dari KMS/Vault | Peringatan. `false` berarti `student_id` tersimpan apa adanya di disk. |
| `QDRANT_MODE` | `server` | Mode `local` adalah basis embedded satu proses — tidak dapat dipakai lebih dari satu worker. |
| `TRUST_PROXY_HEADERS` | `true` **hanya** bila ada proxy tepercaya di depan | `true` tanpa proxy membuat batas laju per IP dapat dilewati dengan mengarang `X-Forwarded-For` tiap permintaan. |
| `ENABLE_ADMIN_API` | `false` kecuali memang dipakai | Bila `true` dan terjangkau publik, satu kunci admin yang bocor membuka **semua** kampus sekaligus. |

Bangkitkan pepper:

```powershell
.venv/Scripts/python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

> **JANGAN mengubah `TENANT_KEY_PEPPER` setelah ada kunci yang terbit.**
> Menggantinya membatalkan seluruh kunci yang beredar di semua kampus sekaligus
> dan memutus verifikasi rantai audit yang sudah tertulis. Tidak ada jalan
> kembali selain menerbitkan ulang semua kunci.

Penjelasan tiap setelan ada di `.env.example`; berkas itu adalah sumber
kebenarannya, bukan tabel di atas.

### 1.2 Pastikan server sehat

```powershell
curl.exe http://localhost:8000/health
```

Di produksi `/docs`, `/redoc`, dan `/openapi.json` sengaja dimatikan, jadi
jangan pakai keduanya sebagai tanda server hidup.

---

## 2. Membuat tenant dan menerbitkan kuncinya

### 2.1 Pilih `tenant_id`

`tenant_id` ikut menjadi nama direktori di disk (`storage/{tenant_id}/`), jadi
polanya ketat: **huruf kecil, angka, strip, garis bawah; 2–63 karakter.** Tanpa
titik dan garis miring.

Nilainya **permanen dalam praktiknya** — ia melekat pada path berkas, payload
Qdrant, kunci cache, dan direktori audit. Mengubahnya di kemudian hari berarti
migrasi data, bukan penggantian nama. Pakai slug institusi yang stabil
(`unair`, `itb`, `poltek-malang`), bukan singkatan program studi.

### 2.2 Buat tenant

```powershell
.venv/Scripts/python.exe -m scripts.tenantctl create unair `
  --name "Universitas Airlangga" `
  --rpm 60 --rpd 5000 --max-upload-mb 100 --max-storage-mb 20000
```

Nilai kuota di atas adalah bawaannya; sebutkan hanya yang berbeda. Kuota adalah
rem biaya: setiap pertanyaan memanggil LLM berbayar, dan satu klien kampus yang
salah membuat perulangan dapat menghabiskan tagihan seluruh layanan.

Keluaran:

```
Tenant dibuat: unair (Universitas Airlangga)
  Kuota: {'requests_per_minute': 60, 'requests_per_day': 5000, ...}

Terbitkan kunci dengan:
  python -m scripts.tenantctl issue-key unair
```

Catatan tenant tersimpan sebagai `data/tenants/unair.json`. Yang ada di dalamnya
hanya **hash** kunci, tidak pernah kuncinya sendiri.

### 2.3 Terbitkan kunci — satu kunci per pemakai, bukan satu per kampus

Hak akses melekat pada **kunci**, bukan pada tenant. Jangan pernah memberikan
satu kunci yang sama kepada aplikasi mahasiswa, sistem akademik, dan dosen:
kunci yang bocor dari aplikasi mahasiswa (yang beredar di ponsel setiap
mahasiswa) akan sekaligus menjadi kunci unggah materi.

Hak yang tersedia (`src/tenancy.py`):

| Hak | Guna |
|---|---|
| `chat:ask` | bertanya, kuis, livecode |
| `catalog:read` | katalog mata kuliah → minggu → materi |
| `content:read` | menelusuri daftar berkas |
| `content:write` | **mengunggah & mengindeks materi** |
| `conversation:read` / `conversation:delete` | riwayat percakapan |
| `answer:validate` | dosen menilai jawaban AI |
| `admin:tenants` | pengelolaan tenant |

**a. Kunci aplikasi mahasiswa** — tanpa `--scopes`, kunci memakai hak pembaca
(`chat:ask`, `catalog:read`, `content:read`, `conversation:read`,
`conversation:delete`). Tidak dapat mengunggah apa pun.

```powershell
.venv/Scripts/python.exe -m scripts.tenantctl issue-key unair --label "App mahasiswa"
```

**b. Kunci sistem akademik (pengunggah)** — satu-satunya yang boleh menulis.
Diberikan ke server LMS/SIAKAD kampus, tidak pernah ke aplikasi klien.

```powershell
.venv/Scripts/python.exe -m scripts.tenantctl issue-key unair `
  --label "SIAKAD - pengunggah" --scopes content:write,content:read,catalog:read
```

**c. Kunci dosen (penilai jawaban)** — boleh menilai dan membaca, tidak boleh
mengunggah.

```powershell
.venv/Scripts/python.exe -m scripts.tenantctl issue-key unair `
  --label "Panel dosen FST" --scopes answer:validate,catalog:read,content:read,chat:ask `
  --courses sbd,kka
```

`--courses` adalah pembatasan ABAC: kunci itu hanya boleh menyentuh mata kuliah
yang disebut. Pakai untuk kunci per fakultas atau per dosen, supaya panel dosen
Fakultas Sains tidak dapat membaca jawaban mahasiswa Fakultas Hukum. Tanpa
`--courses`, kunci mencakup seluruh mata kuliah milik tenant itu.

`--admin` memberikan **seluruh** hak termasuk `admin:tenants`. Kunci ini dapat
menerbitkan kunci untuk kampus mana pun — jangan pernah diberikan kepada
kampus, hanya untuk operator layanan.

Keluaran `issue-key`:

```
Kunci diterbitkan untuk tenant 'unair'
  key_id : 3f9a1c7e5b204d88
  hak    : catalog:read, chat:ask, content:read, ...

  KUNCI (disimpan sekarang juga — tidak ditampilkan lagi):

    ragk_3f9a1c7e5b204d88.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

  Kirimkan lewat kanal yang aman. Server hanya menyimpan hash-nya.
```

> Nilai mentah kunci **hanya muncul sekali**. Kalau hilang: cabut kunci itu
> (bagian 6) dan terbitkan yang baru. Tidak ada cara memulihkannya, dan memang
> itu tujuannya. Kirim lewat kanal yang aman — bukan chat grup, bukan e-mail
> biasa, bukan tiket dukungan.

Periksa hasilnya:

```powershell
.venv/Scripts/python.exe -m scripts.tenantctl list
```

---

## 3. Mengunggah materi

Kunci menentukan tenant. **Tidak ada** parameter yang dapat mengalihkan
unggahan ke kampus lain — `tenant_id` di form body hanya dicocokkan, dan
ketidakcocokan ditolak 403.

### 3.1 Aturan `content_id` — baca sebelum unggahan pertama

`content_id` bukan sekadar nama folder. Ia menentukan **tiga** hal sekaligus,
jadi salah menamainya di awal berarti membongkar ulang katalog kemudian.

**Bentuk yang dianjurkan: `<kode-matkul>-minggu-<n>`** (mis. `sbd-minggu-2`).
Pola lain yang dikenali: `week`/`w`, pemisah `-` `_` `/` — `sbd_minggu_2`,
`algo-w3` sama sahnya (`src/catalog.py`).

Pola karakter yang sah: diawali huruf/angka, lalu huruf, angka, titik, strip,
garis bawah; maksimal 128 karakter; **tanpa** `..` dan tanpa garis miring
sebagai pemisah path.

Tiga akibatnya:

1. **Katalog.** `sbd-minggu-2` → mata kuliah `sbd`, minggu `2`. Inilah yang
   menjadi tombol pilihan "mata kuliah → minggu → materi" di chat. `content_id`
   tanpa pola minggu (mis. `bahan-tambahan`) dibaca sebagai mata kuliah
   **tanpa** minggu, dan tidak akan muncul di penelusuran per minggu.
2. **Penyimpanan.** Berkas mendarat di
   `storage/{tenant_id}/{content_id}/{nama-berkas}`. Dua kampus yang memakai
   `content_id` sama tetap berada di direktori berbeda.
3. **Deteksi berkas rangkap.** Cakupan duplikat adalah **satu direktori
   `content_id`**. Berkas yang isinya sama persis (SHA-256 byte berkas) dengan
   yang sudah ada di `content_id` yang sama akan **ditolak sebelum diproses**.
   Berkas sama di `content_id` berbeda **bukan** duplikat — slide yang dipakai
   ulang di dua minggu memang harus terindeks dua kali, karena kueri mahasiswa
   disaring per minggu.

> **Selalu kirim `content_id`.** Unggahan tanpa `content_id` mendarat di
> `storage/{tenant_id}/_uploads/` dengan awalan UUID. Di sana mata kuliah tidak
> lagi terbaca dari path, sehingga seluruh mata kuliah kampus itu masuk ke satu
> cakupan duplikat yang sama — berkas dua mata kuliah berbeda bisa saling
> dianggap rangkap.

### 3.2 Unggah satu berkas

```powershell
curl.exe -X POST http://localhost:8000/documents/upload `
  -H "X-API-Key: ragk_<kunci-pengunggah>" `
  -F "file=@D:\materi\sbd\pertemuan-2.pdf" `
  -F "content_id=sbd-minggu-2" `
  -F "course_id=sbd" `
  -F "course_name=Sistem Basis Data" `
  -F "week=2"
```

`course_id`, `course_name`, dan `week` opsional — bila kosong, ketiganya
diturunkan dari `content_id`. **Isi `course_name` secara eksplisit pada unggahan
pertama tiap mata kuliah.** Tanpa itu nama yang tampil adalah tebakan mekanis
dari slug (`dasprog-rka` → "Dasprog Rka"), dan nama itulah yang muncul sebagai
tombol pilihan di chat sekaligus ikut masuk ke prompt LLM.

Balasan `201` berisi `chunks_created` dan `points_stored`. `points_stored = 0`
berarti berkas tersimpan tetapi tidak menghasilkan potongan apa pun — periksa
lognya sebelum lanjut.

Penolakan yang wajar ditemui:

| Kode | Arti |
|---|---|
| `400` `content_id tidak valid` | pola nama tidak lolos (lihat 3.1) |
| `400` `Jenis berkas '.x' tidak didukung` | ekstensi di luar daftar putih |
| `400` `Isi '...' sama persis dengan '...'` | duplikat di `content_id` yang sama |
| `403` | kunci tidak punya `content:write`, atau `--courses` tidak mencakup `course_id` |
| `413` | melampaui batas unggah atau kuota simpan tenant |

### 3.3 Indeks ulang seluruh isi satu materi

Dipakai setelah menyalin banyak berkas langsung ke
`storage/{tenant_id}/{content_id}/` di server. Berkas non-dokumen (video, audio,
gambar) dilewati diam-diam bila transkripsi mati.

```powershell
curl.exe -X POST http://localhost:8000/documents/index-batch `
  -H "X-API-Key: ragk_<kunci-pengunggah>" -H "Content-Type: application/json" `
  -d '{\"content_id\":\"sbd-minggu-2\"}'
```

### 3.4 Pastikan katalog kampus sudah terbentuk

```powershell
curl.exe -H "X-API-Key: ragk_<kunci-mahasiswa>" http://localhost:8000/catalog/courses
curl.exe -H "X-API-Key: ragk_<kunci-mahasiswa>" http://localhost:8000/catalog/courses/sbd/weeks
```

Katalog diturunkan dari apa yang **benar-benar terindeks**, jadi mata kuliah
atau minggu yang tidak muncul berarti materinya belum masuk index — bukan
sekadar salah tampil.

---

## 4. Kalender akademik kampus

Bawaannya satu semester 16 minggu: kuis di minggu 4 dan 12, ETS di minggu 8,
EAS di minggu 16. Kampus dengan 14 minggu, atau yang ETS-nya di minggu 7, wajib
menetapkan kalendernya sendiri — kalau tidak, jadwal evaluasi yang disajikan
kepada mahasiswanya salah tanpa ada galat apa pun.

Periksa kalender yang sedang berlaku, persis seperti yang dilihat mahasiswa
kampus itu:

```powershell
curl.exe -H "X-API-Key: ragk_<kunci-mahasiswa>" http://localhost:8000/evaluation/schedule
```

Dari sisi operator, tanpa `--set` sub-perintah berikut hanya menampilkan apa
yang tersimpan — tidak mengubah apa pun:

```powershell
.venv/Scripts/python.exe -m scripts.tenantctl schedule unair
```

```
Jadwal evaluasi 'unair' — bawaan (tenant belum menentukan):
  minggu 4   Kuis                         menguji minggu 1-4       8 soal ...
  minggu 8   Evaluasi Tengah Semester     menguji minggu 1-8       17 soal ...
  minggu 12  Kuis                         menguji minggu 9-12      8 soal ...
  minggu 16  Evaluasi Akhir Semester      menguji minggu 9-16      19 soal ...
```

Baris pertama membedakan **bawaan (tenant belum menentukan)** dari **sendiri**.
Selama masih tertulis "bawaan", kampus itu memakai kalender 4/8/12/16 — bukan
kalendernya.

Tetapkan kalender kampus dengan `--set`. Nilainya daftar dipisah koma, tiap
entri `minggu:jenis:awal-akhir`: jenisnya `kuis`, `ets`, atau `eas`, dan
`awal-akhir` adalah rentang minggu yang diuji — kuis menguji blok terdekat,
sementara ETS/EAS kumulatif atas separuh semester.

```powershell
# Contoh kampus 14 minggu: kuis m3 & m10, ETS m7, EAS m14
.venv/Scripts/python.exe -m scripts.tenantctl schedule unair `
  --set "3:kuis:1-3,7:ets:1-7,10:kuis:8-10,14:eas:8-14"
```

```
Jadwal evaluasi 'unair' — sendiri:
  minggu 3   Kuis                         menguji minggu 1-3       8 soal ...
  minggu 7   Evaluasi Tengah Semester     menguji minggu 1-7       17 soal ...
  minggu 10  Kuis                         menguji minggu 8-10      8 soal ...
  minggu 14  Evaluasi Akhir Semester      menguji minggu 8-14      19 soal ...
```

Tidak ada argumen jumlah minggu: panjang semester tersirat dari entri
terakhirnya. Jadwal cacat ditolak di tempat dengan keluar kode 1 dan tidak
menulis apa pun — misalnya evaluasi yang menguji minggu yang belum berlangsung
(`7:ets:1-9` → `Evaluasi minggu 7 tidak dapat mencakup sampai minggu 9`),
minggu ganda, atau jenis di luar ketiganya. Aturan yang sama dipakai saat berkas
tenant dimuat, jadi berkas yang disunting tangan pun tidak bisa lolos.

`--reset` mengembalikan kampus itu ke bawaan. Medan `evaluation_schedule` di
berkas tenant dikosongkan menjadi `null` — bukan diisi 4/8/12/16 secara
harfiah, sehingga kampus ini ikut berubah bila bawaannya kelak diubah:

```powershell
.venv/Scripts/python.exe -m scripts.tenantctl schedule unair --reset
```

Verifikasi sesudahnya: `GET /evaluation/schedule` memakai kunci kampus itu harus
mengembalikan minggu yang baru ditetapkan, dan kunci kampus lain harus tetap
mengembalikan kalendernya sendiri.

---

## 5. Verifikasi isolasi — jangan dilewati

**Bagian terpenting dalam runbook ini.** Kampus belum boleh dinyatakan siap
sebelum kesembilan langkah di bawah lulus. Isolasi yang tidak pernah diuji sama
saja dengan isolasi yang tidak ada: kegagalannya senyap dan baru ketahuan dari
pihak yang salah.

Siapkan dua kunci pembaca dan satu `content_id` yang **hanya** dimiliki masing-
masing kampus:

```powershell
$BARU  = "ragk_<kunci-pembaca-kampus-baru>"      # mis. unair
$LAMA  = "ragk_<kunci-pembaca-kampus-lain>"      # kampus yang sudah jalan
$CID_LAMA = "sbd-minggu-2"                        # content_id milik kampus lain
$CID_BARU = "basdat-minggu-1"                     # content_id milik kampus baru
```

| # | Langkah | Hasil yang WAJIB |
|---|---|---|
| 1 | `GET /browse/contents` dengan `$BARU` | Hanya `content_id` milik kampus baru. `$CID_LAMA` **tidak** ada dalam daftar. |
| 2 | `GET /browse/contents` dengan `$LAMA` | `$CID_BARU` **tidak** ada dalam daftar. (Arah sebaliknya — sama pentingnya.) |
| 3 | `GET /browse/contents/$CID_LAMA/files` dengan `$BARU` | **404**, bukan 403. |
| 4 | `GET /catalog/courses` dengan `$BARU` | Hanya mata kuliah kampus baru. |
| 5 | `POST /chat/ask` dengan `$BARU`, pertanyaan tentang topik yang **hanya** ada di materi kampus lain | `sources` kosong atau hanya berisi berkas kampus baru. Tidak satu pun `source_file` milik kampus lain. |
| 6 | `POST /chat/ask` dengan `$BARU` dan body `"tenant_id":"<kampus-lain>"` | **403** `tenant_id pada body tidak cocok dengan kredensial`. |
| 7 | `POST /documents/upload` dengan kunci **mahasiswa** | **403** — kunci pembaca tidak boleh menulis. |
| 8 | Permintaan tanpa header `X-API-Key` | **401**. Bila lolos, artinya kunci warisan masih aktif atau tenant belum terdaftar — hentikan dan perbaiki bagian 1. |
| 9 | `GET /catalog/courses` dengan kunci ber-`--courses` | Hanya mata kuliah yang disebut di `--courses`. |

Perintah untuk langkah yang paling sering salah dibaca:

```powershell
# 3 — materi kampus lain harus 404, bukan 403 (403 membocorkan bahwa id itu ada)
curl.exe -i -H "X-API-Key: $BARU" http://localhost:8000/browse/contents/$CID_LAMA/files

# 5 — kebocoran lintas kampus yang paling mungkin: jawaban mengutip materi orang lain
curl.exe -X POST http://localhost:8000/chat/ask `
  -H "X-API-Key: $BARU" -H "Content-Type: application/json" `
  -d '{\"question\":\"<topik yang hanya ada di materi kampus lain>\",\"guided\":false}'

# 6 — body tidak pernah menentukan tenant; ketidakcocokan harus 403
curl.exe -i -X POST http://localhost:8000/chat/ask `
  -H "X-API-Key: $BARU" -H "Content-Type: application/json" `
  -d '{\"question\":\"halo\",\"tenant_id\":\"kampus-lain\",\"guided\":false}'

# 8 — tanpa kunci sama sekali
curl.exe -i http://localhost:8000/catalog/courses
```

Periksa juga lapisan di bawah HTTP:

```powershell
# Berkas benar-benar terpisah: hanya materi kampus itu yang ada di direktorinya
Get-ChildItem storage/unair

# Jejak audit kampus itu utuh (rantai MAC tidak terputus)
.venv/Scripts/python.exe -m scripts.tenantctl verify-audit unair
```

Terakhir, jalankan uji isolasi otomatis. Uji ini menyapu **setiap** metode
publik `QdrantStore` dan memastikan semuanya menolak `tenant_id` kosong,
sehingga metode baru yang lupa menyaring tertangkap tanpa perlu diingat siapa
pun:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_tenancy.py tests/test_regresi_abac.py `
  tests/test_admin_tenant.py -q
```

Satu pun dari langkah 1–9 yang tidak sesuai = **berhenti**. Jangan serahkan
kunci apa pun ke kampus itu sebelum penyebabnya ditemukan.

---

## 6. Mencabut, membekukan, menghapus

Tiga tindakan berbeda dengan akibat yang berbeda. Pilih yang paling ringan yang
masih menyelesaikan masalahnya.

### 6.1 Cabut satu kunci

Untuk kunci yang bocor, atau vendor aplikasi yang berganti. Kampusnya tetap
berjalan dengan kunci-kuncinya yang lain.

```powershell
.venv/Scripts/python.exe -m scripts.tenantctl list                 # cari key_id
.venv/Scripts/python.exe -m scripts.tenantctl revoke unair 3f9a1c7e5b204d88
```

Catatan kuncinya **disimpan**, hanya ditandai dicabut — riwayat siapa memegang
kunci apa tetap utuh untuk audit. Berlaku seketika.

### 6.2 Bekukan seluruh kampus

Untuk tagihan yang belum dibayar atau insiden yang sedang ditangani. Semua
kuncinya berhenti berlaku sekaligus, **tanpa** menghapus data apa pun, dan dapat
dibatalkan.

```powershell
.venv/Scripts/python.exe -m scripts.tenantctl suspend unair
.venv/Scripts/python.exe -m scripts.tenantctl activate unair   # membatalkannya
```

### 6.3 Hapus seluruh data kampus

Langganan berakhir, atau kampus meminta datanya dihapus. **Tidak dapat
dibatalkan.** Butuh `ENABLE_ADMIN_API=true` dan kunci ber-`admin:tenants`.

```powershell
# 1. bekukan dulu, supaya tidak ada unggahan masuk di tengah penghapusan
.venv/Scripts/python.exe -m scripts.tenantctl suspend unair

# 2. hapus data terindeks (Qdrant) dan seluruh percakapan
curl.exe -X DELETE http://localhost:8000/admin/tenants/unair/data `
  -H "X-API-Key: ragk_<kunci-admin>"

# 3. hapus berkas mentahnya — TIDAK ikut terhapus oleh langkah 2
Remove-Item -Recurse -Force storage/unair

# 4. pastikan bersih: katalog kosong dan direktori sudah tidak ada
curl.exe -H "X-API-Key: ragk_<kunci-kampus>" http://localhost:8000/catalog/courses
```

Yang **sengaja tidak** ikut terhapus:

- **Catatan tenant** (`data/tenants/unair.json`) — bukti tenant itu pernah ada,
  siapa saja kuncinya, dan kapan dicabut.
- **Jejak audit** (`data/audit/unair/`) — bukti bahwa penghapusan itu benar
  dilakukan, oleh siapa, dan kapan. Menghapusnya menghapus buktinya sendiri.
- **Berkas di `storage/`** — karena itu langkah 3 harus dikerjakan manual.
  Lewatkan langkah ini dan materi kampus tetap ada di disk setelah datanya
  "dihapus".

Simpan keluaran langkah 2 dan hasil `verify-audit` sebagai bukti penghapusan.

---

## Lampiran — daftar periksa satu halaman

```
[ ] TENANT_KEY_PEPPER terisi, RAGACADEMIC_API_KEY kosong, CORS bukan '*'
[ ] tenantctl create <tenant_id>            (slug institusi, permanen)
[ ] issue-key: mahasiswa  (hak pembaca, tanpa --scopes)
[ ] issue-key: pengunggah (content:write)   -> hanya ke server kampus
[ ] issue-key: dosen      (answer:validate, + --courses bila perlu)
[ ] Unggah materi dengan content_id "<matkul>-minggu-<n>" + course_name eksplisit
[ ] GET /catalog/courses menampilkan mata kuliah kampus itu
[ ] Kalender akademik: tenantctl schedule <id> --set "..." bila bukan 4/8/12/16
[ ] Verifikasi isolasi langkah 1-9 LULUS semua
[ ] pytest tests/test_tenancy.py tests/test_regresi_abac.py hijau
[ ] Kunci dikirim lewat kanal aman; kunci admin TIDAK diberikan ke kampus
```
