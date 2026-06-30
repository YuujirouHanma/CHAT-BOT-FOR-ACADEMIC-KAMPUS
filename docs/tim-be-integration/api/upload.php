<?php
/**
 * upload.php — Upload materi ke RAGAcademic untuk diindex
 *
 * Endpoint ini menerima file dari admin/dosen dan meneruskannya
 * ke RAGAcademic untuk disimpan dan langsung diindex ke vector database.
 * Setelah diindex, materi tersebut bisa dicari lewat chatbot.php.
 *
 * KONFIGURASI YANG HARUS DISESUAIKAN:
 *   - RAG_BASE_URL : URL server RAGAcademic (tanya ke tim RAG)
 *   - RAG_API_KEY  : Key autentikasi (tanya ke tim RAG)
 *   - Hanya role tertentu yang boleh upload (lihat bagian AUTH di bawah)
 *
 * INPUT (multipart/form-data):
 *   - file     : file  — dokumen yang diupload (PDF, DOCX, PPTX, dll)
 *   - class_id : int   — ID kelas dari sistem tim BE
 *
 * OUTPUT (JSON):
 *   - success        : bool
 *   - message        : string
 *   - source_file    : string — nama file yang tersimpan
 *   - chunks_created : int    — jumlah chunk yang diindex
 *   - content_id     : string — identifier yang dipakai RAGAcademic
 */

session_start();

header('Content-Type: application/json; charset=utf-8');

// =====================================================================
// KONFIGURASI — SESUAIKAN SEBELUM DEPLOY
// =====================================================================
define('RAG_BASE_URL', 'http://GANTI_DENGAN_IP_SERVER_RAG:8000');
define('RAG_API_KEY',  'GANTI_DENGAN_API_KEY_DARI_TIM_RAG');
// =====================================================================

// AUTH — hanya dosen/admin yang boleh upload
$role = strtolower((string)($_SESSION['role'] ?? $_SESSION['user_role'] ?? 'guest'));
$allowedRoles = ['dosen', 'admin', 'teacher'];
if (!in_array($role, $allowedRoles, true)) {
    http_response_code(403);
    echo json_encode(['success' => false, 'message' => 'Akses ditolak. Hanya dosen/admin yang bisa upload.']);
    exit;
}

// Validasi request method
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['success' => false, 'message' => 'Method tidak diizinkan.']);
    exit;
}

// Validasi file
if (empty($_FILES['file']) || $_FILES['file']['error'] !== UPLOAD_ERR_OK) {
    $errCode = $_FILES['file']['error'] ?? -1;
    echo json_encode(['success' => false, 'message' => "Upload gagal. Error code: $errCode"]);
    exit;
}

$classId = (int)($_POST['class_id'] ?? 0);

// Mapping: class_id → content_id (harus konsisten dengan chatbot.php)
$contentId = $classId > 0 ? 'class-' . $classId : null;

// Siapkan multipart/form-data untuk RAGAcademic
$filePath = $_FILES['file']['tmp_name'];
$fileName = basename($_FILES['file']['name']);
$mimeType = $_FILES['file']['type'] ?: 'application/octet-stream';

$postFields = [
    'file' => new CURLFile($filePath, $mimeType, $fileName),
];
if ($contentId !== null) {
    $postFields['content_id'] = $contentId;
}

$ch = curl_init(RAG_BASE_URL . '/documents/upload');
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POST           => true,
    CURLOPT_POSTFIELDS     => $postFields,
    CURLOPT_HTTPHEADER     => [
        'X-API-Key: ' . RAG_API_KEY,
        // Content-Type otomatis multipart/form-data saat POSTFIELDS array + CURLFile
    ],
    CURLOPT_TIMEOUT        => 300, // Indexing bisa lama untuk dokumen besar
]);

$rawResponse = curl_exec($ch);
$httpCode    = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$curlError   = curl_error($ch);
curl_close($ch);

if ($curlError || !in_array($httpCode, [200, 201])) {
    http_response_code(502);
    echo json_encode([
        'success' => false,
        'message' => 'Gagal mengindex dokumen ke RAGAcademic.',
        'error'   => $curlError ?: "HTTP $httpCode",
    ]);
    exit;
}

$ragResponse = json_decode($rawResponse, true);

echo json_encode([
    'success'        => true,
    'message'        => 'Dokumen berhasil diupload dan diindex.',
    'source_file'    => $ragResponse['source_file']    ?? $fileName,
    'chunks_created' => $ragResponse['chunks_created'] ?? 0,
    'points_stored'  => $ragResponse['points_stored']  ?? 0,
    'content_id'     => $ragResponse['content_id']     ?? $contentId,
]);
