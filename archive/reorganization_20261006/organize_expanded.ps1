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
$manifestPath = Join-Path $PSScriptRoot 'expanded_manifest.json'
if (Test-Path -LiteralPath $manifestPath) { throw 'A migration manifest already exists; do not repeat this migration.' }
$profile = Get-Content -LiteralPath (SafePath 'configs/deployment.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$active = @($profile.models.PSObject.Properties | ForEach-Object { $_.Value.run } | Sort-Object -Unique)
$suite = Get-Content -LiteralPath (SafePath 'reports/expanded_v6/suite/status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($suite.status -ne 'complete') { throw 'Wait for the frozen v6 suite to finish' }
$gate = Get-Content -LiteralPath (SafePath 'reports/expanded_v6/deployment_gate.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($gate.decision -ne 'retain_existing_v4_deployment' -or $profile.dataset -ne 'data/processed/text_only_v4_shared') { throw 'Expected the recorded decision to retain stable v4' }
$moves = @()
$eligible = @($suite.completed) + @('expanded_v5_classical_s42','expanded_v5_classical_s43','expanded_v5_classical_s44','expanded_v5_mbert_last_s42','source_holdout_v1_classical_s42','source_holdout_v1_kaz_roberta_last_s42','train_length_ablation_v1_classical_s42','train_length_ablation_v1_kaz_roberta_last_s42','text_only_v4_classical_s42','text_only_v4_mbert_last_s42','text_only_v4_kaz_roberta_last_s42','text_only_v4_kaz_roberta_concat4_s42')
foreach ($name in $eligible) {
    if ($active -contains $name) { continue }
    $source = SafePath ('artifacts/'+$name)
    if (-not (Test-Path -LiteralPath $source)) { throw "Expected declared artifact: $name" }
    if ($name -ne 'expanded_v5_mbert_last_s42' -and -not (Test-Path -LiteralPath (Join-Path $source 'results.json'))) { throw "Incomplete run: $name" }
    $moves += [PSCustomObject]@{source='artifacts/'+$name; destination='archive/artifacts/'+$name; reason='Inactive or intentionally aborted run; preserve weights, predictions and provenance'}
}
$oldReports = @('expanded_v5_suite','experiment_text_only_v4','text_only_v4_repeated','text_only_v4_diagnostics','text_only_v4_cohorts','text_only_v4_figures','text_only_v4_linear_audit','text_only_v4_user_diagnostics','web_probe_v4_20261006','dissertation_draft_20261006_morphology','morphology_ablation_v1')
foreach ($name in $oldReports) {
    $moves += [PSCustomObject]@{source='reports/'+$name; destination='archive/reports/'+$name; reason='Historical v4 evidence; the active study is reports/expanded_v6'}
}
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
[System.IO.File]::WriteAllText((Join-Path $PSScriptRoot 'expanded_verification.json'), ($result | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
$result | ConvertTo-Json
