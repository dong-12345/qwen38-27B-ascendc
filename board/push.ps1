# push.ps1 - upload local text files to the board (LF-normalized, base64 via stdin)
# usage (from pwsh): & .\board\push.ps1 -Map @("code\qwen38_310b\ops\kernels\x.cpp=~/code/qwen38_310B/src/kernels/x.cpp", ...)
param([Parameter(Mandatory=$true)][string[]]$Map)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$key  = Join-Path $root "tools\opi_key"
$board = "HwHiAiUser@192.168.137.163"
foreach ($m in $Map) {
    $p = $m.Split("=")
    $local  = Join-Path $root $p[0]
    $remote = $p[1]
    $txt = (Get-Content -Path $local -Raw) -replace "`r", ""
    $b64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($txt))
    $cmd = "mkdir -p `$(dirname $remote); base64 -d > $remote; wc -c $remote"
    $r = $b64 | & ssh -i $key -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -o LogLevel=ERROR $board $cmd
    Write-Host ("push {0} -> {1} : {2}" -f $p[0], $remote, ($r -join " "))
}
