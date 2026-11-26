# spush.ps1 - upload local files to the board with scp (binary safe), with retry
# usage: & .\board\spush.ps1 -Map @("code\qwen38_310b\ops\kernels\x.cpp=~/code/qwen38_310B/src/kernels/x.cpp", ...)
param(
  [Parameter(Mandatory=$true)][string[]]$Map,
  [int]$Retries = 5,
  [int]$RetryDelaySec = 8
)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$key  = Join-Path $root "tools\opi_key"
$board = "HwHiAiUser@192.168.137.163"
function Invoke-WithRetry([scriptblock]$Action, [string]$What) {
  for ($i = 1; $i -le $Retries; $i++) {
    $out = & $Action 2>&1
    $txt = ($out | Out-String)
    if ($txt -match "Connection timed out|Connection reset|Connection refused|Broken pipe|No route to host|Connection closed by") {
      Write-Host "[spush] $What flaky, retry $i/$Retries"
      Start-Sleep -Seconds $RetryDelaySec
      continue
    }
    return $out
  }
  throw "[spush] $What unreachable after $Retries attempts"
}
foreach ($m in $Map) {
    $p = $m.Split("=")
    $local  = Join-Path $root $p[0]
    $remote = $p[1]
    $null = Invoke-WithRetry { & ssh -i $key -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -o LogLevel=ERROR -o ConnectTimeout=12 $board "mkdir -p `$(dirname $remote)" } "mkdir"
    $null = Invoke-WithRetry { & scp -i $key -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -o LogLevel=ERROR -o ConnectTimeout=12 $local "${board}:$remote" } "scp"
    $r = Invoke-WithRetry { & ssh -i $key -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -o LogLevel=ERROR -o ConnectTimeout=12 $board "md5sum $remote; wc -c $remote" } "md5"
    Write-Host ("push {0} -> {1} : {2}" -f $p[0], $remote, ($r -join " "))
}