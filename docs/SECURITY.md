# Keamanan & Multi-Tenancy RAGAcademic

**Untuk:** tim AI, tim Back-End, dan siapa pun yang menyiapkan deployment
**Versi layanan:** 0.4.0
**Pola tenancy:** satu basis data / satu koleksi, dibedakan kolom `tenant_id`
(*shared schema, discriminator column*)

---

## 0. Ringkas: apa yang berubah dan mengapa

Sampai versi 0.3.0 layanan ini menganggap hanya ada satu pemakai selamanya: satu
koleksi Qdrant, satu folder penyimpanan, satu API key global. Begitu dijual ke
kampus kedua, asumsi itu berubah menjadi kebocoran data — katalog, hasil
pencarian, dan riwayat percakapan semua pelanggan bercampur dalam satu ruang.

Versi 0.4.0 menutup itu. Aturan pusatnya satu kalimat:

> **`tenant_id` hanya lahir dari kunci API, tidak pernah dari body permintaan.**

Semua yang ada di bawah ini adalah turunan dari aturan itu.

---

## 1. Model isolasi

| Lapisan | Cara memisahkan | Berkas |
|---|---|---|
| Vektor (Qdrant) | Field payload `tenant_id` + filter **wajib** di setiap kueri | `src/storage/qdrant_store.py` |
| Berkas materi | Direktori terpisah `storage/{tenant_id}/{content_id}/` | `src/storage/content_store.py` |
| Riwayat percakapan | Direktori terpisah `data/conversations/{tenant_id}/` | `src/storage/conversation_store.py` |
| Cache hasil LLM | `tenant_id` masuk kunci hash | `src/gen_cache.py` |
| Session di memori | Cache berkunci `(tenant_id, session_id)` | `src/api/session.py` |
| Log HITL | Direktori terpisah `data/hitl_logs/{tenant_id}/` | `src/hitl/logger.py` |
| Jejak audit | Direktori terpisah `data/audit/{tenant_id}/` | `src/security/audit.py` |

### Kenapa cara memisahkannya berbeda-beda

**Qdrant memakai filter, bukan satu koleksi per tenant.** Setiap koleksi Qdrant
membawa biaya tetap (segmen, indeks HNSW, memori) yang tidak masuk akal
dikalikan jumlah pelanggan, dan menambah pelanggan baru jadi menuntut operasi
administratif. Qdrant sendiri menganjurkan pola payload + indeks bertanda tenant
(`is_tenant=True`), yang juga menata data satu tenant berdekatan di disk sehingga
kuerinya lebih cepat.

**Berkas dipisah secara fisik, bukan disaring saat dibaca.** Dua pelanggan yang
memakai `content_id` sama — dan pola `<matkul>-minggu-<n>` memang dianjurkan,
jadi tabrakan pasti terjadi — akan saling menimpa berkas kalau ditaruh di satu
direktori. Penyaringan saat baca tidak menolong: kerusakannya sudah terjadi pada
saat penulisan.

### Dua penjagaan struktural

Isolasi ini tidak bergantung pada kedisiplinan pemanggil:

1. **`tenant_id` adalah argumen wajib** (keyword-only, tanpa nilai bawaan) pada
   setiap metode penyimpanan. Lupa mengirimnya adalah `TypeError` saat
   pemanggilan — bukan pencarian diam-diam ke seluruh korpus.
2. **`QdrantStore._build_filter()` selalu mengembalikan filter berisi
   `tenant_id`, dan tidak pernah `None`.** Versi lama mengembalikan `None` bila
   tak ada penyaring, yang di Qdrant berarti "cari di seluruh koleksi". Jalur
   itu sudah tidak ada.

Keduanya diuji di `tests/test_tenancy.py`, termasuk uji sapuan yang memastikan
**setiap** metode publik `QdrantStore` menolak `tenant_id` kosong — sehingga
metode baru yang lupa menyaring tertangkap tanpa perlu diingat siapa pun.

---

## 2. Autentikasi & otorisasi

### Bentuk kunci

```
ragk_<key_id>.<secret>
     └─ 16 hex   └─ 256 bit acak (base64url)
```

`key_id` bukan rahasia. Ia dipakai mencari catatan kunci secara langsung (O(1))
dan aman ditulis ke log untuk menelusuri kunci mana yang dipakai atau harus
dicabut. Nama tenant sengaja **tidak** disertakan di dalam kunci — kunci sering
tidak sengaja tersalin ke tiket dukungan dan log CI, dan nama pelanggan tidak
perlu ikut bocor bersamanya.

### Kenapa HMAC-SHA256, bukan Argon2id, untuk kunci API

Argon2id dirancang untuk rahasia beruntropi **rendah** — kata sandi buatan
manusia yang bisa ditebak dari kamus; biaya komputasinya yang tinggi itulah
pertahanannya. Kunci di sini 256 bit acak: menebaknya mustahil terlepas dari
fungsi hash apa pun, sementara Argon2id akan menambah ±100 ms pada **setiap**
permintaan API — beban nyata tanpa manfaat.

Karena itu dipakai HMAC-SHA256 dengan *pepper* sisi server, praktik yang sama
dipakai penyedia API besar. *Pepper* disimpan di luar basis data, jadi bocornya
salinan basis data saja belum cukup untuk memakai kunci-kunci di dalamnya.

Untuk kata sandi manusia — bila nanti ada login dosen/admin —
`src/security/keys.py` menyediakan `hash_password()` yang memakai **Argon2id**
(atau PBKDF2-HMAC-SHA256 600.000 iterasi bila `argon2-cffi` belum terpasang).

### RBAC + ABAC di lapisan domain

Hak melekat pada **kunci**, bukan pada tenant — satu kampus boleh punya kunci
unggah untuk sistem akademiknya dan kunci baca-saja untuk aplikasi mahasiswanya.

| Hak | Guna |
|---|---|
| `chat:ask` | bertanya, menjalankan kuis |
| `catalog:read` | membaca katalog mata kuliah → minggu → materi |
| `content:read` | menelusuri daftar berkas |
| `content:write` | mengunggah & mengindeks materi |
| `conversation:read` / `conversation:delete` | riwayat percakapan |
| `admin:tenants` | pengelolaan tenant |

**ABAC:** sebuah kunci dapat dibatasi pada mata kuliah tertentu
(`allowed_courses`) — berguna untuk kampus yang ingin satu kunci per fakultas.
Pemeriksaan dilakukan di rute (lapisan domain), bukan hanya di gerbang API,
sehingga rute baru yang lupa memasangnya gagal karena tidak punya tenant sama
sekali — bukan diam-diam terbuka untuk semua kunci.

### Pencegahan IDOR/BOLA

Setiap akses ke satu catatan diperiksa terhadap tenant pemanggil:

- Percakapan milik tenant lain dibalas **404, bukan 403.** Membedakan keduanya
  memberi tahu penyerang bahwa sebuah `conversation_id` memang ada — cukup untuk
  memetakan id yang valid lewat percobaan berulang.
- `session_id` bolak-balik lewat klien, jadi diperlakukan sebagai nilai yang bisa
  dicuri: mengirim `session_id` milik tenant lain menghasilkan session **baru**,
  bukan session orang lain.
- `tenant_id` di body **dicocokkan** dengan yang diturunkan dari kunci;
  ketidakcocokan → 403 + catatan audit. Nilai di body tidak pernah menentukan
  cakupan data.

### Kunci global warisan

`RAGACADEMIC_API_KEY` (satu kunci untuk semua pemanggil) masih dikenali **di luar
produksi** agar integrasi tim BE tidak putus di hari peralihan. Di produksi,
server **menolak start** bila kunci itu masih terisi — kunci yang sama untuk
semua pemanggil tidak dapat membedakan siapa pun, sehingga isolasi tidak mungkin
ditegakkan untuk permintaan yang memakainya.

---

## 3. Kriptografi & perlindungan data

### Data pribadi

Data pribadi di sistem ini pada dasarnya satu: **`student_id`** — ia merangkai
seluruh riwayat belajar satu orang. Ia tidak disimpan apa adanya, melainkan
sebagai dua turunan berbeda dari nilai yang sama:

| Turunan | Algoritma | Guna |
|---|---|---|
| `student_id` (tersimpan) | AES-256-GCM, nonce acak, *envelope encryption* | kerahasiaan |
| `student_ref` | HMAC-SHA256, deterministik | penyaringan tanpa dekripsi |

Enkripsi deterministik untuk keduanya sekaligus akan tampak lebih sederhana,
tetapi membocorkan kesamaan nilai (dua baris dengan sandi sama = orang yang sama)
dan membuka analisis frekuensi. Karena itu sengaja dipisah.

`tenant_id` menjadi *additional authenticated data*: sandi milik tenant A yang
dipindahkan ke berkas tenant B akan **gagal didekripsi**, bukan diam-diam
terbaca. `tenant_id` juga ikut masuk masukan HMAC indeks buta, sehingga NIM yang
sama di dua kampus menghasilkan indeks yang berbeda — tanpa itu, dua tenant dapat
saling menyimpulkan bahwa seseorang terdaftar di keduanya.

### Envelope encryption & KMS

Setiap nilai disandikan dengan kunci data (DEK) acak; DEK itu sendiri disandikan
dengan kunci induk (KEK). Dampaknya: memutar kunci induk **tidak** menuntut
penyandian ulang seluruh basis data, dan kunci induk tidak pernah hadir di dalam
kode maupun berkas konfigurasi.

`EnvKeyProvider` (KEK dari variabel lingkungan) **hanya untuk pengembangan** —
kunci induk berakhir di berkas `.env`, log proses, dan citra kontainer. Di
produksi ganti dengan penyedia KMS; antarmuka `KeyProvider` hanya dua metode
(`wrap`/`unwrap`) supaya AWS KMS, Google Cloud KMS, atau Vault Transit cukup
mengimplementasikan keduanya tanpa menyentuh satu pun pemanggil.

### Nilai lama tetap terbaca

`decrypt_field()` mengembalikan nilai yang belum tersandi apa adanya. Penyandian
karenanya dapat dinyalakan pada data yang sudah ada tanpa migrasi serentak; baris
lama tetap terbaca sampai tersentuh penulisan berikutnya.

---

## 4. Validasi masukan & pencegahan serangan

| Serangan | Penjagaan |
|---|---|
| Path traversal | `tenant_id`, `content_id`, `session_id` dibatasi pola ketat (tanpa `/`, `\`, `..`); path hasilnya diperiksa ulang dengan `resolve()` agar benar-benar berada di dalam direktori tenant |
| Traversal lewat nama berkas | `resolve_file()` **menolak** nama bermuatan pemisah path, bukan menormalkannya — permintaan `../bab1.pdf` yang diam-diam dilayani berkas lain menyembunyikan klien bermasalah maupun klien yang sedang menjajaki batas |
| Field tak dikenal | `extra="forbid"` pada seluruh skema permintaan |
| Muatan berlebihan | Batas ukuran body di middleware; unggahan dibatasi **sambil ditulis**, bukan sesudahnya — `Content-Length` dikirim klien dan bisa berbohong |
| Jenis berkas | Daftar putih ekstensi |
| Injeksi SQL | Tidak berlaku — tidak ada SQL; Qdrant diakses lewat objek filter berjenis, bukan string kueri |
| XSS | API hanya membalas JSON; CSP `default-src 'none'` menahan skrip pada balasan HTML yang lolos (mis. halaman galat proxy) |
| Clickjacking | `X-Frame-Options: DENY` + `frame-ancestors 'none'` |
| MIME sniffing | `X-Content-Type-Options: nosniff` |
| Host palsu | `TrustedHostMiddleware` |

**Header keamanan** (`src/api/middleware.py`): CSP, `X-Frame-Options`,
`X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`,
`Cross-Origin-Opener/Resource-Policy`, `Cache-Control: no-store`, dan HSTS di
produksi.

**CORS**: `allow_credentials` hanya dinyalakan bila daftar asal sudah spesifik.
Bintang bersama kredensial membuat halaman mana pun di internet dapat memanggil
API ini memakai kredensial pengunjungnya — di produksi kombinasi itu
menggagalkan startup.

**SSRF**: satu-satunya permintaan keluar dari layanan ini adalah ke penyedia LLM
dan transkripsi, yang alamatnya berasal dari konfigurasi server
(`PROVIDER_BASE_URLS`), bukan dari masukan pengguna. Tidak ada endpoint "ambil
URL ini". Bila nanti ditambahkan (misalnya impor materi dari tautan), ia
**wajib** memakai daftar putih host beserta validasi DNS terhadap rentang alamat
internal.

---

## 5. Pembatasan laju & anti-penyalahgunaan

Tiga lapis, dievaluasi dari yang paling murah (`src/security/ratelimit.py`):

1. **Per alamat IP** — menahan banjir permintaan sebelum kredensial diperiksa,
   termasuk lalu lintas yang tidak membawa kunci sama sekali
2. **Per kunci API** — satu integrasi yang salah membuat perulangan tidak
   menghabiskan jatah seluruh tenant
3. **Per tenant (menit & hari)** — batas komersial sesungguhnya; inilah yang
   menjaga tagihan LLM tetap terkendali dan menjadi dasar penjualan berjenjang

Algoritma *token bucket*: membolehkan lonjakan wajar (mahasiswa membuka riwayat
lalu langsung bertanya) tanpa melonggarkan laju rata-rata, dan tanpa masalah
"reset serentak di detik ke-0" seperti jendela tetap.

**Anti-penebakan kunci** (`AuthThrottle`): penundaan naik berlipat (1s, 2s, 4s, …)
hingga batas atas, lalu ditolak langsung. Dihitung dari **alamat IP**, bukan
`key_id` — penebak justru mencoba `key_id` yang berbeda-beda, jadi membatasi per
`key_id` tidak akan pernah menyentuhnya. Kegagalan dilupakan setelah jendela
tenang berlalu, sehingga salah konfigurasi sesaat tidak mengunci integrasi yang
sah selamanya.

> ### ⚠️ Batasan penting
> Hitungannya disimpan **di memori proses**. Dengan lebih dari satu worker
> uvicorn atau lebih dari satu replika, tiap proses punya hitungan sendiri —
> batas efektif menjadi berlipat jumlah proses. **Untuk produksi wajib diganti
> backend Redis** (skrip Lua `INCR`+`EXPIRE` atau `CL.THROTTLE`); antarmuka
> `_Backend` sengaja dibuat sempit agar penggantinya tidak menyentuh pemanggil.

---

## 6. Log, audit, dan penanganan galat

### Balasan galat tidak membocorkan isi dalam sistem

Semua galat dibalas dengan bentuk seragam:

```json
{ "error": "kode_stabil", "detail": "pesan singkat", "request_id": "a1b2c3…" }
```

Galat 5xx **selalu** dibalas pesan umum; rinciannya hanya masuk log server.
Jejak tumpukan menyebutkan path berkas di server, versi pustaka, dan kadang
potongan kueri beserta datanya — peta gratis bagi siapa pun yang sedang menjajaki
sistem. Galat validasi menyebut **field** yang salah tanpa memantulkan kembali
nilainya, karena nilai itu bisa saja berisi kredensial yang salah tempat.

`request_id` ada di setiap balasan galat dan setiap catatan audit, sehingga
pengguna cukup melaporkan satu kode pendek dan operator menemukan barisnya di log
tanpa perlu rincian apa pun ikut terkirim keluar.

**Dokumentasi interaktif (`/docs`, `/redoc`, `/openapi.json`) dimatikan di
produksi** — nilainya bagi tim integrasi tidak sebanding dengan peta permukaan
API yang diberikannya kepada siapa pun yang menemukan alamat layanan ini.
`/health` juga tidak lagi menyebut versi dan nama model.

### Jejak audit yang perubahannya terdeteksi

Setiap peristiwa penting dicatat sebagai satu baris JSON yang memuat MAC dari
isinya digabung MAC baris sebelumnya. Mengubah atau menghapus satu baris memutus
seluruh MAC sesudahnya. Rantai memakai **HMAC**-SHA256, bukan SHA-256 biasa:
dengan hash biasa, siapa pun yang bisa menulis berkasnya juga bisa menghitung
ulang seluruh rantai setelah mengubah isinya — jejaknya tampak utuh padahal sudah
dipalsukan.

Peristiwa yang dicatat: `auth.success`, `auth.failure`, `auth.forbidden`,
`auth.tenant_mismatch`, `auth.rate_limited`, `content.upload`, `content.index`,
`conversation.read`, `conversation.delete`, `chat.ask`.

Verifikasi: `python -m scripts.tenantctl verify-audit <tenant_id>`

> **Batasan jujur:** ini membuat perubahan **terdeteksi**, bukan **mustahil**.
> Penyerang yang menguasai server tetap bisa menghapus berkasnya. Untuk
> kepatuhan sungguhan, alirkan juga ke penyimpanan yang hanya bisa ditulis
> sekali (S3 Object Lock, CloudWatch Logs, atau SIEM).

### Penyuntingan log

Dilakukan di sisi **penulis** (`src/security/redact.py`), bukan diserahkan pada
kedisiplinan tiap pemanggil. Log adalah tempat rahasia paling sering bocor: ia
disalin ke tiket dukungan, dikirim ke layanan agregasi pihak ketiga, dan disimpan
jauh lebih lama daripada data aslinya.

- Kunci & token (`ragk_…`, `Bearer …`, `sk-…`, `gsk_…`, `hf_…`) → diganti topeng
- Field bernama sensitif (`password`, `secret`, `token`, `authorization`, …) → topeng
- Data pribadi (`student_id`, surel, NIM) → disamarkan sebagian (`20…34`) — cukup
  untuk mencocokkan dua baris log tentang orang yang sama saat menelusuri
  masalah, tidak cukup untuk mengetahui siapa orangnya dari log saja

---

## 7. Menyiapkan tenant

```bash
# 1. Buat tenant beserta kuotanya
python -m scripts.tenantctl create kampus-a --name "Universitas A" \
    --rpm 120 --rpd 20000 --max-upload-mb 200 --max-storage-mb 50000

# 2. Kunci untuk sistem akademik (boleh mengunggah materi)
python -m scripts.tenantctl issue-key kampus-a --label "BE produksi" --admin

# 3. Kunci untuk aplikasi mahasiswa (baca & tanya saja)
python -m scripts.tenantctl issue-key kampus-a --label "Aplikasi mahasiswa"

# 4. Kunci yang dibatasi satu fakultas (ABAC)
python -m scripts.tenantctl issue-key kampus-a --label "Fakultas Ilkom" \
    --courses sbd,kka

python -m scripts.tenantctl list
python -m scripts.tenantctl revoke kampus-a <key_id>
python -m scripts.tenantctl suspend kampus-a       # hentikan pelanggan menunggak
```

Nilai mentah kunci **hanya ditampilkan sekali**. Kalau hilang: cabut kunci itu,
terbitkan yang baru. Tidak ada cara memulihkannya — memang itu tujuannya.

---

## 8. Konfigurasi produksi

Server **menolak start** di `APP_ENV=production` bila ada butir bertanda ❌.

| Setelan | Nilai produksi | Wajib |
|---|---|---|
| `APP_ENV` | `production` | — |
| `TENANT_KEY_PEPPER` | 32+ byte acak, dari secret manager | ❌ |
| Tenant terdaftar | minimal satu | ❌ |
| `RAGACADEMIC_API_KEY` | **kosong** (kunci warisan) | ❌ |
| `CORS_ALLOW_ORIGINS` | daftar asal spesifik, bukan `*` | ❌ |
| `ENCRYPT_PII` + `PII_KEK` | `true` + KEK dari KMS | ❌ bila `true` tanpa `cryptography` |
| `TRUSTED_HOSTS` | nama host layanan | ⚠️ |
| `TRUST_PROXY_HEADERS` | `true` **hanya** bila ada proxy tepercaya di depan | ⚠️ |
| `ENABLE_HSTS` | `true` | ⚠️ |

> `TENANT_KEY_PEPPER` **tidak boleh berubah** setelah ada kunci terbit:
> menggantinya membuat seluruh kunci yang beredar tidak lagi cocok dan memutus
> verifikasi rantai audit yang sudah tertulis.

`TRUST_PROXY_HEADERS` sengaja mati secara bawaan. Tanpa proxy tepercaya di depan
yang menimpa header itu, `X-Forwarded-For` sepenuhnya dikendalikan pemanggil —
mempercayainya membuat pembatasan laju per IP dapat dilewati hanya dengan
mengarang nilai baru setiap permintaan.

---

## 9. Yang HARUS disiapkan di infrastruktur (bukan di kode ini)

Butir berikut ada di daftar persyaratan tetapi **tidak dapat dipenuhi oleh kode
aplikasi**. Didokumentasikan di sini agar tidak dianggap sudah selesai.

| Kebutuhan | Status | Cara memenuhi |
|---|---|---|
| **TLS 1.3 + cipher kuat** | infrastruktur | Terminasi TLS di ingress/ALB/nginx. Aplikasi mengirim HSTS; ia tidak menyediakan TLS sendiri |
| **mTLS antar layanan** | infrastruktur | Service mesh (Istio/Linkerd) atau sidecar Envoy. BE ↔ RAG idealnya mTLS, bukan hanya kunci API |
| **Segmentasi jaringan** | infrastruktur | Qdrant **tidak boleh** terjangkau dari internet — subnet privat + security group. Qdrant tidak punya autentikasi bawaan; jaringanlah satu-satunya pelindungnya |
| **KMS / Vault** | infrastruktur + 1 kelas | Implementasikan `KeyProvider` untuk AWS KMS / GCP KMS / Vault Transit. Antarmukanya sudah ada, penyedianya belum |
| **Redis untuk pembatasan laju** | infrastruktur + 1 kelas | Ganti `_MemoryBackend`. **Wajib** bila jalan lebih dari satu worker/replika |
| **WAF** | infrastruktur | CloudFront / Cloudflare / ALB WAF di depan layanan |
| **Log sekali-tulis (WORM)** | infrastruktur | Alirkan `data/audit/` ke S3 Object Lock atau SIEM |
| **Enkripsi disk (at rest)** | infrastruktur | EBS/PD terenkripsi untuk volume Qdrant & `storage/` — enkripsi field hanya melindungi PII, bukan isi materi |
| **OAuth 2.1 / OIDC + PKCE** | **tidak berlaku** | Lihat catatan di bawah |
| **Cookie HttpOnly / SameSite** | **tidak berlaku** | Lihat catatan di bawah |

### Kenapa OAuth 2.1/PKCE dan cookie tidak diterapkan di sini

Layanan ini adalah **API antar-layanan**: yang memanggilnya adalah back-end
kampus, bukan peramban mahasiswa. OAuth 2.1 dengan PKCE dirancang untuk klien
publik yang tidak dapat menyimpan rahasia (aplikasi ponsel, SPA) dan berpusat
pada alur persetujuan pengguna — tidak satu pun berlaku pada pemanggil
server-ke-server yang memang bisa menyimpan rahasia dengan aman. Begitu pula
cookie `HttpOnly`/`SameSite`: tidak ada peramban dalam alur ini, jadi tidak ada
cookie yang perlu diamankan.

Yang tepat untuk pola ini adalah **kunci API panjang, diacak, ter-hash, dan dapat
dicabut satu per satu** — dan itulah yang diterapkan.

**Autentikasi mahasiswa tetap tanggung jawab tim BE.** Layanan ini tidak mengenal
mahasiswa; ia hanya menerima `student_id` sebagai label. Kalau nanti mahasiswa
perlu login langsung ke layanan ini, barulah OIDC + token akses berumur pendek +
refresh token berputar menjadi relevan — dan `hash_password()` berbasis Argon2id
sudah disiapkan untuk saat itu.

---

## 10. Menguji isolasi

```bash
.venv/Scripts/python.exe -m pytest tests/test_tenancy.py -q
```

123 uji, dua tenant palsu (`kampus-a`, `kampus-b`) yang sengaja diberi data
saling mirip — nama mata kuliah sama, `content_id` sama, `student_id` sama —
karena tabrakan nama adalah bentuk kebocoran yang paling mungkin terjadi di dunia
nyata.

`TestQdrantIsolation` memakai **Qdrant sungguhan** dalam mode memori, bukan
tiruan: menirukan penyimpanan berarti menirukan pula penyaringnya, dan penyaring
itulah yang sedang diuji.

**Jalankan uji ini di CI pada setiap perubahan.** Nilainya bukan pada saat
migrasi ini, melainkan enam bulan lagi — ketika seseorang menambah endpoint baru
dan lupa menyaring tenant, dan uji inilah yang memberi tahu, bukan laporan
pelanggan yang melihat data kampus lain.
