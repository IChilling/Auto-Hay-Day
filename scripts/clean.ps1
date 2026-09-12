<#
.SYNOPSIS
Remove generated Python caches from this checkout.
.DESCRIPTION
Use -IncludeBuilds to also remove root build/ and dist/ outputs.
Use -WhatIf to preview targets. Reference images, artifacts, the virtual
environment, and application data are retained. Directory links are skipped.
#>
[CmdletBinding(SupportsShouldProcess)]
param([switch]$IncludeBuilds)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\', '/')
$projectPrefix = $projectRoot + [IO.Path]::DirectorySeparatorChar
$cacheNames = @('__pycache__', '.pytest_cache', '.ruff_cache')
$skipNames = @('.venv', '.git', '.local-data', 'artifacts', 'build', 'dist')
$pending = [Collections.Generic.Stack[string]]::new()
$targets = [Collections.Generic.List[string]]::new()
$pending.Push($projectRoot)

while ($pending.Count -gt 0) {
    foreach ($directory in Get-ChildItem -LiteralPath $pending.Pop() -Directory -Force) {
        if ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            continue
        }
        if ($directory.Name -in $cacheNames) {
            $targets.Add($directory.FullName)
        } elseif ($directory.Name -notin $skipNames) {
            $pending.Push($directory.FullName)
        }
    }
}

if ($IncludeBuilds) {
    foreach ($name in @('build', 'dist')) {
        $candidate = Join-Path $projectRoot $name
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            $targets.Add($candidate)
        }
    }
}

$removedCount = 0
[long]$removedBytes = 0
foreach ($target in $targets) {
    $resolved = (Resolve-Path -LiteralPath $target).ProviderPath
    $absolute = [IO.Path]::GetFullPath($resolved)
    if (-not $absolute.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup target is outside the project: $absolute"
    }
    $item = Get-Item -LiteralPath $absolute -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        Write-Warning "Skipping directory link: $absolute"
        continue
    }
    $children = @(Get-ChildItem -LiteralPath $absolute -Recurse -Force)
    if (@($children | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        Write-Warning "Skipping target containing a directory link: $absolute"
        continue
    }
    $size = ($children | Where-Object { -not $_.PSIsContainer } |
        Measure-Object -Property Length -Sum).Sum
    if ($PSCmdlet.ShouldProcess($absolute, 'Remove generated files')) {
        Remove-Item -LiteralPath $absolute -Recurse -Force
        $removedCount++
        $removedBytes += $size
    }
}

Write-Output ('Removed {0} generated directories ({1:N2} MiB).' -f
    $removedCount, ($removedBytes / 1MB))
