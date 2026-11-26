# brun.ps1 - run a local bash script on the board via ssh (base64, so no CRLF/quoting issues)
# usage: powershell -File board\brun.ps1 -Script board\xxx.sh
# auto-retry on flaky links (board is on a Windows hotspot and drops periodically)
param(
  [Parameter(Mandatory=$true)][string]$Script,
  [int]$Retries = 6,
  [int]$RetryDelaySec = 8
)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$key  = Join-Path $root "tools\opi_key"
$board = "HwHiAiUser@192.168.137.163"
$body = (Get-Content -Path (Join-Path $root $Script) -Raw) -replace "`r", ""
$b64  = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($body))
$remote = "echo $b64 | base64 -d | bash"
for ($i = 1; $i -le $Retries; $i++) {
  $out = & ssh -i $key -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o LogLevel=ERROR $board $remote 2>&1
  $txt = ($out | Out-String)
  if ($txt -match "Connection timed out|Connection reset|Connection refused|Broken pipe|No route to host|Connection closed by|closed by remote host") {
    Write-Host "[brun] link flaky, retry $i/$Retries in ${RetryDelaySec}s"
    Start-Sleep -Seconds $RetryDelaySec
    continue
  }
  $out
  exit $LASTEXITCODE
}
Write-Host "[brun] board unreachable after $Retries attempts"
exit 2