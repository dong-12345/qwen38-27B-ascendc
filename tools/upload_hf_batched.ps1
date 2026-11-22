# upload_hf_batched.ps1 -- upload converted W4A16 weights to HuggingFace in batches.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/upload_hf_batched.ps1
#
# Why batched: a single 16.5GB upload stalls on flaky networks; per-layer batches are
# resumable (files already on the Hub are skipped by sha256) and failures stay local.
# Uses hf-mirror + disables Xet (the Xet token endpoint is often blocked in CN).
param(
    [string]$Repo = "dong2542107715/Qwen3.8-27B-AscendC-W4A16",
    [string]$Root = "converted\qwen38_310b",
    [string]$Log  = "tmp\hf_upload.log"
)
$ErrorActionPreference = "Continue"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
$hf = "$env:USERPROFILE\.local\bin\hf.exe"

function Write-Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) | Out-Null

$batch = New-Object System.Collections.ArrayList
foreach ($f in @("meta.txt", "LICENSE", "MODIFICATIONS.md", "manifest.json")) {
    $p = Join-Path $Root $f
    if (Test-Path $p) { [void]$batch.Add(@{ kind = "file"; local = $p; remote = $f; name = $f }) }
}
foreach ($d in @("model\language_model\embed_tokens", "model\language_model\norm", "mtp", "lm_head")) {
    $p = Join-Path $Root $d
    if (Test-Path $p) { [void]$batch.Add(@{ kind = "dir"; local = $p; remote = $d.Replace("\", "/"); name = $d }) }
}
for ($i = 0; $i -lt 64; $i++) {
    $rel = "model\language_model\layers\$i"
    $p = Join-Path $Root $rel
    if (Test-Path $p) { [void]$batch.Add(@{ kind = "dir"; local = $p; remote = "model/language_model/layers/$i"; name = "layers/$i" }) }
}

Write-Log ("START repo={0} batches={1}" -f $Repo, $batch.Count)
$okCount = 0
$failList = New-Object System.Collections.ArrayList
$swAll = [System.Diagnostics.Stopwatch]::StartNew()

for ($bi = 0; $bi -lt $batch.Count; $bi++) {
    $b = $batch[$bi]
    $tag = "[{0}/{1}] {2}" -f ($bi + 1), $batch.Count, $b.name
    $done = $false
    for ($try = 1; $try -le 3; $try++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $hf upload $Repo $b.local $b.remote --commit-message ("upload " + $b.remote) *>&1 | Out-Null
        $code = $LASTEXITCODE
        $sw.Stop()
        if ($code -eq 0) {
            Write-Log ("{0} OK ({1:N1}s)" -f $tag, $sw.Elapsed.TotalSeconds)
            $okCount++
            $done = $true
            break
        }
        Write-Log ("{0} FAIL try {1} (exit={2}, {3:N1}s)" -f $tag, $try, $code, $sw.Elapsed.TotalSeconds)
        Start-Sleep -Seconds 5
    }
    if (-not $done) { [void]$failList.Add($b.name) }
}

$swAll.Stop()
Write-Log ("DONE ok={0} fail={1} elapsed={2:N1}min" -f $okCount, $failList.Count, $swAll.Elapsed.TotalMinutes)
if ($failList.Count -gt 0) { Write-Log ("FAILED BATCHES: " + ($failList -join ", ")) }
