# push_converted_weights.ps1 — 把转换后的 27B 权重（19.5GB）逐文件推到开发板（可断点续传）
param(
    [string]$Board = "192.168.137.163",
    [string]$Key = "D:\项目4\orangepi_aipro\tools\opi_key",
    [string]$Local = "D:\项目4\orangepi_aipro\converted\qwen38_310b",
    [string]$Remote = "/home/HwHiAiUser/models/qwen38_310b"
)
$ErrorActionPreference = 'Continue'
$sshOpts = @("-i", $Key, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
             "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30")

# 远端建目录
& ssh @sshOpts "HwHiAiUser@$Board" "mkdir -p $Remote" 2>&1 | Out-Null

$files = Get-ChildItem $Local -Recurse -File | Sort-Object Length -Descending
$total = ($files | Measure-Object Length -Sum).Sum
Write-Output ("files: {0}, total {1:N2} GB" -f $files.Count, ($total / 1GB))

$sent = 0
$sentBytes = 0
foreach ($f in $files) {
    $rel = $f.FullName.Substring($Local.Length + 1).Replace('\', '/')
    $dest = "$Remote/$rel"
    $remoteDir = ($dest -replace '/[^/]+$', '')
    $size = $f.Length
    # 检查远端是否已有且大小一致
    if ($sent % 20 -eq 0) { & ssh @sshOpts "HwHiAiUser@$Board" "mkdir -p '$remoteDir'" 2>&1 | Out-Null }
    $remoteSize = & ssh @sshOpts "HwHiAiUser@$Board" "stat -c %s '$dest' 2>/dev/null || echo 0" 2>&1
    if ($remoteSize -and [int64]$remoteSize -eq $size) {
        $sent++; $sentBytes += $size
        continue
    }
    & scp @sshOpts $f.FullName "HwHiAiUser@${Board}:$dest" 2>&1 | Out-Null
    $sent++; $sentBytes += $size
    if ($sent % 10 -eq 0) {
        Write-Output ("[{0}/{1}] {2:N2}/ {3:N2} GB  ({4})" -f $sent, $files.Count, ($sentBytes / 1GB), ($total / 1GB), $rel)
    }
}
Write-Output ("DONE: {0}/{1} files, {2:N2} GB" -f $sent, $files.Count, ($sentBytes / 1GB))
