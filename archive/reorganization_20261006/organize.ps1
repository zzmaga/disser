param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$prefix = $root.TrimEnd('\') + '\'
function SafePath([string]$relative) {
    $absolute = [System.IO.Path]::GetFullPath((Join-Path $root $relative))
    if (-not $absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes project: $relative"
    }
    return $absolute
}
$manifestPath = Join-Path $PSScriptRoot 'manifest.json'
if (Test-Path -LiteralPath $manifestPath) { throw 'A migration manifest already exists; do not repeat this migration.' }
$profile = Get-Content -LiteralPath (SafePath 'configs/deployment.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$active = @($profile.models.PSObject.Properties | ForEach-Object { $_.Value.run } | Sort-Object -Unique)
$moves = @()
Get-ChildItem -LiteralPath (SafePath 'artifacts') -Directory | ForEach-Object {
    if ($active -notcontains $_.Name -and $_.Name -match '^(pilot_v2_|text_only_v3_|text_only_v4_|gpu_pipeline_check_)') {
        $moves += [PSCustomObject]@{source='artifacts/'+$_.Name; destination='archive/artifacts/'+$_.Name; reason='Inactive checkpoint; retain experiment evidence'}
    }
}
$oldReports = @('data_audit_v2','dissertation_draft_20261006','dissertation_draft_20261006_results','pilot_v2','pilot_v2_diagnostics','text_only_v3','text_only_v3_cohorts','text_only_v3_diagnostics','web_probe_20261006')
foreach ($name in $oldReports) {
    $moves += [PSCustomObject]@{source='reports/'+$name; destination='archive/reports/'+$name; reason='Superseded report or draft'}
}
Get-ChildItem -LiteralPath (SafePath 'reports') -File | Where-Object { $_.Extension -ne '.html' -and $_.Name -ne 'README.md' } | ForEach-Object {
    $moves += [PSCustomObject]@{source='reports/'+$_.Name; destination='reports/technical/'+$_.Name; reason='Technical evidence or log; keep out of the report entry point'}
}
$moves += [PSCustomObject]@{source='docs/research'; destination='archive/docs/research'; reason='Historical notes superseded by the current project passport'}
$moves += [PSCustomObject]@{source='docs/ARCHITECTURE.md'; destination='archive/docs/ARCHITECTURE_before_cleanup.md'; reason='Architecture consolidated into current workflow'}
# Preflight every path and collision before the first mutation.
foreach ($move in $moves) {
    $source = SafePath $move.source
    $destination = SafePath $move.destination
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing source: $source" }
    if (Test-Path -LiteralPath $destination) { throw "Existing destination: $destination" }
    $item = Get-Item -LiteralPath $source
    if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Refusing a junction/symlink: $source" }
}
if (-not $Apply) { $moves | Format-Table -AutoSize; return }
$files = @()
foreach ($move in $moves) {
    $source = SafePath $move.source
    $item = Get-Item -LiteralPath $source
    $members = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $source -Recurse -File) } else { @($item) }
    foreach ($file in $members) {
        if ($file.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Refusing linked file: $($file.FullName)" }
        $suffix = if ($item.PSIsContainer) { $file.FullName.Substring($source.Length).Replace('\','/') } else { '' }
        $files += [PSCustomObject]@{source=$move.source+$suffix; destination=$move.destination+$suffix; bytes=$file.Length; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
    }
}
$record = @{created_at=[DateTime]::UtcNow.ToString('o'); active_runs=$active; deployment_sha256=(Get-FileHash -LiteralPath (SafePath 'configs/deployment.json') -Algorithm SHA256).Hash; moves=$moves; files=$files; deletions=0}
[System.IO.File]::WriteAllText($manifestPath, ($record | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
foreach ($move in $moves) {
    $source = SafePath $move.source
    $destination = SafePath $move.destination
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($destination)) | Out-Null
    Move-Item -LiteralPath $source -Destination $destination
}
foreach ($file in $files) {
    $destination = SafePath $file.destination
    if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) {
        throw "Hash verification failed after move: $destination"
    }
}
$result = @{status='verified'; files=$files.Count; moves=$moves.Count; bytes=($files | Measure-Object bytes -Sum).Sum; deleted=0}
[System.IO.File]::WriteAllText((Join-Path $PSScriptRoot 'verification.json'), ($result | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
$result | ConvertTo-Json
