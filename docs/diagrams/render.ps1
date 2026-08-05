# Render semua diagram Mermaid menjadi SVG (vektor, untuk naskah) dan PNG (pratinjau).
#
#   powershell -ExecutionPolicy Bypass -File docs\diagrams\render.ps1
#
# SVG dipakai untuk LaTeX/Word setelah dikonversi ke PDF (lihat README.md).
# PNG di-render pada skala 3x sehingga cukup tajam untuk pratinjau dan slide.

$ErrorActionPreference = "Stop"

$here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcDir  = Join-Path $here "src"
$svgDir  = Join-Path $here "svg"
$pngDir  = Join-Path $here "png"
$cfg     = Join-Path $here "mermaid-config.json"
$pptrCfg = Join-Path $here "puppeteer-config.json"

New-Item -ItemType Directory -Force $svgDir | Out-Null
New-Item -ItemType Directory -Force $pngDir | Out-Null

# mermaid-cli merender lewat peramban headless. Daripada mengunduh Chromium
# terpisah (~150 MB), pakai Chrome/Edge yang sudah ada di sistem.
if (-not $env:PUPPETEER_EXECUTABLE_PATH) {
    $browsers = @(
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
        "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe",
        "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
    )
    $found = $browsers | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($found) {
        $env:PUPPETEER_EXECUTABLE_PATH = $found
        Write-Host "Peramban: $found" -ForegroundColor DarkGray
    } else {
        Write-Host "Chrome/Edge tidak ditemukan. Jalankan: npx puppeteer browsers install chrome" -ForegroundColor Yellow
    }
}

# Pakai mmdc lokal bila variabel MMDC diset, kalau tidak ambil lewat npx.
if ($env:MMDC -and (Test-Path $env:MMDC)) {
    $runner = { param($argv) & $env:MMDC @argv }
} else {
    $runner = { param($argv) & npx -y @mermaid-js/mermaid-cli @argv }
}

$files = Get-ChildItem -Path $srcDir -Filter *.mmd | Sort-Object Name
Write-Host "Merender $($files.Count) diagram..." -ForegroundColor Cyan

foreach ($f in $files) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)

    $svg = Join-Path $svgDir "$name.svg"
    & $runner @(@("-i", $f.FullName, "-o", $svg, "-c", $cfg, "-p", $pptrCfg, "-b", "white"))
    if ($LASTEXITCODE -ne 0) { Write-Host "  GAGAL (svg): $name" -ForegroundColor Red; continue }

    $png = Join-Path $pngDir "$name.png"
    & $runner @(@("-i", $f.FullName, "-o", $png, "-c", $cfg, "-p", $pptrCfg, "-b", "white", "-s", "3"))
    if ($LASTEXITCODE -ne 0) { Write-Host "  GAGAL (png): $name" -ForegroundColor Red; continue }

    Write-Host "  OK  $name" -ForegroundColor Green
}

Write-Host "`nSelesai. SVG: $svgDir" -ForegroundColor Cyan
Write-Host "         PNG: $pngDir" -ForegroundColor Cyan
