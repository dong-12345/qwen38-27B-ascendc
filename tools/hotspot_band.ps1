# Switch the mobile hotspot to 5 GHz (or back to 2.4) via WinRT, with COM polling fallback.
param([string]$Band = 'FiveGigahertz')
$ErrorActionPreference = 'Continue'
$SSID = 'DONG'
$KEY = '88888888'

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]

function Await2($op, $type) {
    try {
        $m = $asTaskGeneric.MakeGenericMethod($type)
        $t = $m.Invoke($null, @($op))
        $t.Wait(-1) | Out-Null
        return $t.Result
    } catch {
        Write-Output ("  (AsTask failed: " + $_.Exception.Message + ") - polling")
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            try { $st = $op.Status } catch { $st = $null }
            if ($null -eq $st -or "$st" -eq 'Completed' -or "$st" -eq '1') {
                try { return $op.GetResults() } catch { Write-Output "  GetResults failed: $($_.Exception.Message)"; return $null }
            }
            Start-Sleep -Milliseconds 500
        }
        return $null
    }
}

[Windows.Networking.Connectivity.NetworkInformation, Windows.Networking.Connectivity, ContentType = WindowsRuntime] | Out-Null
[Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager, Windows.Networking.NetworkOperators, ContentType = WindowsRuntime] | Out-Null
[Windows.Networking.NetworkOperators.NetworkOperatorTetheringAccessPointConfiguration, Windows.Networking.NetworkOperators, ContentType = WindowsRuntime] | Out-Null

$resType = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringOperationResult]
$cfgType = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringAccessPointConfiguration]
$bandType = [Windows.Networking.NetworkOperators.TetheringWiFiBand]

$conn = [Windows.Networking.Connectivity.NetworkInformation]::GetInternetConnectionProfile()
$mgr = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager]::CreateFromConnectionProfile($conn)
Write-Output "state: $($mgr.TetheringOperationalState)"

if ($mgr.TetheringOperationalState -eq [Windows.Networking.NetworkOperators.TetheringOperationalState]::On) {
    $r = Await2 ($mgr.StopTetheringAsync()) $resType
    Write-Output "stop: $($r.Status)"
    Start-Sleep -Seconds 2
}
try {
    $cfg = $cfgType::new()
    $cfg.Ssid = $SSID
    $cfg.Passphrase = $KEY
    if ($Band -eq 'FiveGigahertz') {
        $cfg.Band = $bandType::FiveGigahertz
    } else {
        $cfg.Band = $bandType::TwoPointFourGigahertz
    }
    $r = Await2 ($mgr.ConfigureAccessPointAsync($cfg)) $resType
    Write-Output "configure: $($r.Status) $($r.AdditionalErrorMessage)"
} catch { Write-Output "configure-error: $($_.Exception.Message)" }

Start-Sleep -Seconds 1
$r = Await2 ($mgr.StartTetheringAsync()) $resType
Write-Output "start: $($r.Status) $($r.AdditionalErrorMessage)"

Start-Sleep -Seconds 4
$cur = $mgr.GetCurrentAccessPointConfiguration()
Write-Output "now: state=$($mgr.TetheringOperationalState) ssid=$($cur.Ssid) band=$($cur.Band)"
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like '192.168.137.*' } |
    Select-Object InterfaceAlias, IPAddress | Format-Table -AutoSize | Out-String | Write-Output
