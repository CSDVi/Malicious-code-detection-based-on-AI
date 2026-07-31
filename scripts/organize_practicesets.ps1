param(
    [string]$PracticesetsRoot = (Join-Path (Split-Path $PSScriptRoot -Parent) "practicesets"),
    [switch]$WhatIfOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

function Get-FullPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath($Path).TrimEnd("\")
}

function Assert-ChildPath {
    param(
        [string]$Candidate,
        [string]$Root
    )
    $fullCandidate = Get-FullPath $Candidate
    $fullRoot = Get-FullPath $Root
    $prefix = $fullRoot + "\"
    if (-not $fullCandidate.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside practicesets: $fullCandidate"
    }
}

function New-Mapping {
    param(
        [string]$Name,
        [string]$Category,
        [string]$Language
    )
    [PSCustomObject]@{
        Name = $Name
        Category = $Category
        Language = $Language
    }
}

$rootPath = Get-FullPath (Resolve-Path -LiteralPath $PracticesetsRoot).Path
if ([System.IO.Path]::GetFileName($rootPath) -ne "practicesets") {
    throw "Expected a directory named practicesets, got: $rootPath"
}

$vulnerabilityRoot = Join-Path $rootPath "vulnerability_detection"
$malwareRoot = Join-Path $rootPath "malware_detection"
$languageRoots = @{
    "java" = Join-Path $malwareRoot "java"
    "javascript" = Join-Path $malwareRoot "javascript"
    "php" = Join-Path $malwareRoot "php"
    "python" = Join-Path $malwareRoot "python"
    "other" = Join-Path $malwareRoot "other"
}

$mappings = @(
    New-Mapping "2022-05-12-php-test-suite-sqli-v1-0-0" "vulnerability" ""
    New-Mapping "2022-05-12-php-test-suite-sqli-v1-0-0.zip" "vulnerability" ""
    New-Mapping "BenchmarkJava-master" "vulnerability" ""
    New-Mapping "BenchmarkJava-master.zip" "vulnerability" ""
    New-Mapping "crossvul" "vulnerability" ""
    New-Mapping "cwe-bench-java-master" "vulnerability" ""
    New-Mapping "github_advisory" "vulnerability" ""
    New-Mapping "morefixes_v4" "vulnerability" ""
    New-Mapping "patch-files2026-06-20.zip" "vulnerability" ""
    New-Mapping "Vul4J-main" "vulnerability" ""

    New-Mapping "android_malware_java" "malware" "java"
    New-Mapping "android-malware-source-code-samples-main.zip" "malware" "java"

    New-Mapping "javascript_malware_collection" "malware" "javascript"
    New-Mapping "javascript-malware-collection-master.zip" "malware" "javascript"
    New-Mapping "npm" "malware" "javascript"
    New-Mapping "npm_static_extracted" "malware" "javascript"
    New-Mapping "npm_static_extracted_manifest.json" "malware" "javascript"
    New-Mapping "npm_zip.zip" "malware" "javascript"

    New-Mapping "php_webshell" "malware" "php"

    New-Mapping "compromised_lib" "malware" "python"
    New-Mapping "malicious-software-packages-dataset-samples-pypi-malicious_intent" "malware" "python"
    New-Mapping "malicious-software-packages-dataset-samples-pypi.zip" "malware" "python"
    New-Mapping "pypi_compromised_static" "malware" "python"
    New-Mapping "pypi_compromised_static_manifest.json" "malware" "python"
    New-Mapping "pypi_malicious_intent_static_extracted" "malware" "python"
    New-Mapping "pypi_malicious_intent_static_extracted_manifest.json" "malware" "python"
    New-Mapping "pypi_malicious_intent_static_v2" "malware" "python"
    New-Mapping "pypi_malicious_intent_static_v2_manifest.json" "malware" "python"
    New-Mapping "pypi_malregistry_selected" "malware" "python"
    New-Mapping "pypi_malregistry_static" "malware" "python"
    New-Mapping "pypi_malregistry_static_manifest.json" "malware" "python"

    New-Mapping "CodeSearchNet" "malware" "other"
    New-Mapping "datadog_malicious" "malware" "other"
    New-Mapping "manifests" "malware" "other"
    New-Mapping "mascot_metadata" "malware" "other"
    New-Mapping "paired_clean_archives" "malware" "other"
    New-Mapping "paired_clean_manifest.json" "malware" "other"
    New-Mapping "paired_clean_static" "malware" "other"
    New-Mapping "paired_clean_static_manifest.json" "malware" "other"
    New-Mapping "xgboost_multilingual" "malware" "other"
)

$knownNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($mapping in $mappings) {
    [void]$knownNames.Add($mapping.Name)
}
[void]$knownNames.Add("vulnerability_detection")
[void]$knownNames.Add("malware_detection")

$unknown = @(
    Get-ChildItem -LiteralPath $rootPath -Force |
        Where-Object { -not $knownNames.Contains($_.Name) } |
        Select-Object -ExpandProperty FullName
)
if ($unknown.Count -gt 0) {
    throw "Unmapped top-level entries found; no files were moved:`n$($unknown -join "`n")"
}

$records = @()
foreach ($mapping in $mappings) {
    $destinationRoot = if ($mapping.Category -eq "vulnerability") {
        $vulnerabilityRoot
    } else {
        $languageRoots[$mapping.Language]
    }
    $source = Join-Path $rootPath $mapping.Name
    $destination = Join-Path $destinationRoot $mapping.Name
    Assert-ChildPath $source $rootPath
    Assert-ChildPath $destination $rootPath
    $sourceExists = Test-Path -LiteralPath $source
    $destinationExists = Test-Path -LiteralPath $destination
    if ($sourceExists -and $destinationExists) {
        throw "Move conflict: both source and destination exist for $($mapping.Name)"
    }
    if (-not $sourceExists -and -not $destinationExists) {
        throw "Expected dataset entry is missing from both locations: $($mapping.Name)"
    }

    $currentPath = if ($sourceExists) { $source } else { $destination }
    $item = Get-Item -LiteralPath $currentPath -Force
    $sha256 = $null
    if (-not $item.PSIsContainer -and $item.Length -le 128MB) {
        $sha256 = (Get-FileHash -LiteralPath $currentPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $records += [PSCustomObject]@{
        name = $mapping.Name
        category = $mapping.Category
        language = if ($mapping.Language) { $mapping.Language } else { $null }
        source = $source
        destination = $destination
        kind = if ($item.PSIsContainer) { "directory" } else { "file" }
        bytes = if ($item.PSIsContainer) { $null } else { $item.Length }
        sha256 = $sha256
        status = if ($sourceExists) { "pending" } else { "already_moved" }
    }
}

if ($WhatIfOnly) {
    [PSCustomObject]@{
        practicesets_root = $rootPath
        operations = $records
    } | ConvertTo-Json -Depth 5
    exit 0
}

foreach ($directory in @($vulnerabilityRoot, $malwareRoot) + $languageRoots.Values) {
    Assert-ChildPath $directory $rootPath
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

foreach ($record in $records) {
    if ($record.status -eq "pending") {
        Move-Item -LiteralPath $record.source -Destination $record.destination
        $record.status = "moved"
    }
}

function Update-EmbeddedRoot {
    param(
        [string]$ManifestPath,
        [string]$NewRoot
    )
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        return
    }
    $oldJsonRoot = $rootPath.Replace("\", "\\")
    $newJsonRoot = $NewRoot.Replace("\", "\\")
    $rawBytes = [System.IO.File]::ReadAllBytes($ManifestPath)
    $hasUtf8Bom = (
        $rawBytes.Length -ge 3 -and
        $rawBytes[0] -eq 0xEF -and
        $rawBytes[1] -eq 0xBB -and
        $rawBytes[2] -eq 0xBF
    )
    $content = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8
    $duplicateJsonRoot = $newJsonRoot + $newJsonRoot.Substring($oldJsonRoot.Length)
    $updated = $content
    while ($updated.Contains($duplicateJsonRoot)) {
        $updated = $updated.Replace($duplicateJsonRoot, $newJsonRoot)
    }
    $legacyPattern = [regex]::Escape($oldJsonRoot) + '(?!\\\\(?:malware_detection|vulnerability_detection)(?:\\\\|"))'
    $updated = [regex]::Replace($updated, $legacyPattern, $newJsonRoot)
    if ($updated -ne $content -or $hasUtf8Bom) {
        [System.IO.File]::WriteAllText($ManifestPath, $updated, $utf8NoBom)
    }
}

Update-EmbeddedRoot (Join-Path $languageRoots["javascript"] "npm_static_extracted_manifest.json") $languageRoots["javascript"]
foreach ($name in @(
    "pypi_compromised_static_manifest.json",
    "pypi_malicious_intent_static_extracted_manifest.json",
    "pypi_malicious_intent_static_v2_manifest.json",
    "pypi_malregistry_static_manifest.json"
)) {
    Update-EmbeddedRoot (Join-Path $languageRoots["python"] $name) $languageRoots["python"]
}
Update-EmbeddedRoot (Join-Path $languageRoots["other"] "paired_clean_manifest.json") $languageRoots["other"]
Update-EmbeddedRoot (Join-Path $languageRoots["other"] "paired_clean_static_manifest.json") $languageRoots["other"]
Update-EmbeddedRoot (Join-Path $languageRoots["other"] "manifests\pypi_malregistry_selection.json") $languageRoots["python"]

foreach ($record in $records) {
    if ($record.kind -eq "file") {
        $item = Get-Item -LiteralPath $record.destination -Force
        $record.bytes = $item.Length
        $record.sha256 = if ($item.Length -le 128MB) {
            (Get-FileHash -LiteralPath $record.destination -Algorithm SHA256).Hash.ToLowerInvariant()
        } else {
            $null
        }
    }
}

$topLevelAfter = @(Get-ChildItem -LiteralPath $rootPath -Force | Select-Object -ExpandProperty Name | Sort-Object)
if (($topLevelAfter -join "|") -ne "malware_detection|vulnerability_detection") {
    throw "Post-move validation failed: practicesets contains unexpected top-level entries."
}
foreach ($record in $records) {
    if (-not (Test-Path -LiteralPath $record.destination)) {
        throw "Post-move validation failed: $($record.destination)"
    }
}

$manifest = [PSCustomObject]@{
    schema_version = 1
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    practicesets_root = $rootPath
    safety = [PSCustomObject]@{
        deleted_entries = 0
        samples_executed = $false
        operation = "same-volume move"
        hashes_recorded_for_files_up_to_bytes = 128MB
    }
    layout = [PSCustomObject]@{
        vulnerability = $vulnerabilityRoot
        malware = $malwareRoot
        malware_languages = @($languageRoots.Keys | Sort-Object)
    }
    operations = $records
}
$manifestPath = Join-Path $languageRoots["other"] "practicesets_organization_manifest.json"
[System.IO.File]::WriteAllText(
    $manifestPath,
    ($manifest | ConvertTo-Json -Depth 6),
    $utf8NoBom
)

[PSCustomObject]@{
    manifest = $manifestPath
    moved = @($records | Where-Object status -eq "moved").Count
    already_moved = @($records | Where-Object status -eq "already_moved").Count
    top_level = $topLevelAfter
} | ConvertTo-Json -Depth 4
