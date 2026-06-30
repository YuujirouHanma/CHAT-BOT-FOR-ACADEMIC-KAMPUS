<?php
/**
 * chatbot.php — Integrasi RAGAcademic ke sistem tim BE
 *
 * Endpoint ini menerima pertanyaan dari frontend dan meneruskannya
 * ke RAGAcademic (backend RAG berbasis Python/FastAPI), lalu
 * mengembalikan jawaban ke frontend.
 *
 * KONFIGURASI YANG HARUS DISESUAIKAN:
 *   - RAG_BASE_URL : URL server RAGAcademic (tanya ke tim RAG)
 *   - RAG_API_KEY  : Key autentikasi (tanya ke tim RAG, jangan hardcode di sini)
 *
 * INPUT (POST body JSON atau form):
 *   - message   : string — pertanyaan mahasiswa
 *   - class_id  : int    — ID kelas dari sistem tim BE
 *
 * OUTPUT (JSON):
 *   - reply           : string — jawaban AI
 *   - recommendations : array  — pertanyaan lanjutan yang disarankan
 *   - sources         : array  — sumber materi yang dirujuk
 *   - class_id        : int
 *   - role            : string
 */

session_start();

header('Content-Type: application/json; charset=utf-8');

// =====================================================================
// KONFIGURASI — SESUAIKAN SEBELUM DEPLOY
// =====================================================================
define('RAG_BASE_URL', 'http://GANTI_DENGAN_IP_SERVER_RAG:8000');
define('RAG_API_KEY',  'GANTI_DENGAN_API_KEY_DARI_TIM_RAG');
// =====================================================================

$rawInput = file_get_contents('php://input') ?: '';
$payload  = json_decode($rawInput, true);
if (!is_array($payload)) {
    $payload = $_POST;
}

$message = trim((string)($payload['message'] ?? ''));
$classId = (int)($_GET['class_id'] ?? $payload['class_id'] ?? 0);
$role    = strtolower((string)($_SESSION['role'] ?? $_SESSION['user_role'] ?? 'guest'));

if ($message === '') {
    echo json_encode([
        'reply'    => 'Kirim pesan untuk mulai.',
        'class_id' => $classId,
        'role'     => $role,
    ]);
    exit;
}

// Ambil session_id RAGAcademic dari PHP session (untuk multi-turn chat)
$ragSessionId = $_SESSION['rag_session_id'] ?? null;

// Mapping: class_id (int) → content_id (string) untuk RAGAcademic
// Format ini harus konsisten dengan saat file diupload ke RAGAcademic.
// Ubah sesuai kesepakatan dengan tim RAG.
$contentId = $classId > 0 ? 'class-' . $classId : null;

// Panggil RAGAcademic /chat/ask
$requestBody = json_encode([
    'question'   => $message,
    'content_id' => $contentId,
    'session_id' => $ragSessionId,
]);

$ch = curl_init(RAG_BASE_URL . '/chat/ask');
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POST           => true,
    CURLOPT_POSTFIELDS     => $requestBody,
    CURLOPT_HTTPHEADER     => [
        'Content-Type: application/json',
        'X-API-Key: ' . RAG_API_KEY,
    ],
    CURLOPT_TIMEOUT        => 60, // LLM butuh waktu, jangan kurang dari 30 detik
]);

$rawResponse = curl_exec($ch);
$httpCode    = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$curlError   = curl_error($ch);
curl_close($ch);

if ($curlError || $httpCode !== 200) {
    http_response_code(502);
    echo json_encode([
        'reply'    => 'Layanan AI sedang tidak tersedia. Coba lagi nanti.',
        'class_id' => $classId,
        'role'     => $role,
        'error'    => $curlError ?: "HTTP $httpCode",
    ]);
    exit;
}

$ragResponse = json_decode($rawResponse, true);

// Simpan session_id RAGAcademic ke PHP session supaya percakapan nyambung
if (!empty($ragResponse['session_id'])) {
    $_SESSION['rag_session_id'] = $ragResponse['session_id'];
}

echo json_encode([
    'reply'           => $ragResponse['answer']          ?? 'Tidak ada jawaban.',
    'recommendations' => $ragResponse['recommendations'] ?? [],
    'sources'         => $ragResponse['sources']         ?? [],
    'class_id'        => $classId,
    'role'            => $role,
]);
