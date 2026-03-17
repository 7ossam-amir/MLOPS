param(
    [string]$Kernel = "hossamamir/mlops-a2-gpu",
    [string]$KaggleExe = "$env:CONDA_PREFIX\Scripts\kaggle.exe",
    [string]$RunsRoot = ".\kaggle_runs",
    [string]$ArchiveRoot = ".\mlruns_archive"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

function Convert-ToFileUri([string]$Path) {
    $resolved = (Resolve-Path $Path).Path
    return ([System.Uri]$resolved).AbsoluteUri
}

if (-not (Test-Path $KaggleExe)) {
    throw "Kaggle CLI not found at: $KaggleExe"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $RunsRoot $timestamp
New-Item -ItemType Directory -Force $runDir | Out-Null

Write-Host "Downloading kernel outputs to: $runDir"
$supportsFilePattern = $false
$helpText = & $KaggleExe kernels output --help 2>&1
if ($helpText -match "--file-pattern") {
    $supportsFilePattern = $true
}

if ($supportsFilePattern) {
    & $KaggleExe kernels output $Kernel -p $runDir --force --file-pattern "mlops-a2-gpu\.log|outputs/.*"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pattern download failed; retrying full output download..."
        & $KaggleExe kernels output $Kernel -p $runDir --force
    }
} else {
    Write-Host "This Kaggle CLI version does not support --file-pattern; downloading full outputs..."
    & $KaggleExe kernels output $Kernel -p $runDir --force
}
$downloadExitCode = $LASTEXITCODE

$srcMlruns = Join-Path $runDir "outputs\mlruns"
if (-not (Test-Path $srcMlruns)) {
    if ($downloadExitCode -ne 0) {
        throw "Failed to download kernel outputs for $Kernel (exit code: $downloadExitCode)"
    }
    Write-Host "No MLflow store found in this run output. Nothing to archive."
    Write-Host "Run folder kept at: $runDir"
    exit 0
}

if ($downloadExitCode -ne 0) {
    Write-Host "Warning: kaggle output command returned non-zero exit code ($downloadExitCode), but mlruns data exists. Continuing archive..."
}

New-Item -ItemType Directory -Force $ArchiveRoot | Out-Null

Get-ChildItem $srcMlruns -Directory | ForEach-Object {
    $expName = $_.Name
    if ($expName -eq ".trash") {
        return
    }

    $srcExp = $_.FullName
    $dstExp = Join-Path $ArchiveRoot $expName
    New-Item -ItemType Directory -Force $dstExp | Out-Null

    $srcMeta = Join-Path $srcExp "meta.yaml"
    if (Test-Path $srcMeta) {
        Copy-Item $srcMeta (Join-Path $dstExp "meta.yaml") -Force
    }

    Get-ChildItem $srcExp -Directory | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $dstExp $_.Name) -Recurse -Force
    }

    # Rewrite experiment artifact location to local archive path.
    $dstExpMeta = Join-Path $dstExp "meta.yaml"
    if (Test-Path $dstExpMeta) {
        $expUri = Convert-ToFileUri $dstExp
        $metaText = Get-Content $dstExpMeta -Raw
        $metaText = $metaText -replace '(?m)^artifact_location:.*$', "artifact_location: $expUri"
        Set-Content $dstExpMeta $metaText
    }

    # Rewrite each run artifact URI to local archive artifacts folder.
    Get-ChildItem $dstExp -Directory | ForEach-Object {
        $runMeta = Join-Path $_.FullName "meta.yaml"
        $runArtifacts = Join-Path $_.FullName "artifacts"
        if ((Test-Path $runMeta) -and (Test-Path $runArtifacts)) {
            $runUri = Convert-ToFileUri $runArtifacts
            $runText = Get-Content $runMeta -Raw
            $runText = $runText -replace '(?m)^artifact_uri:.*$', "artifact_uri: $runUri"
            Set-Content $runMeta $runText
        }
    }
}

Write-Host "Archived MLflow runs into: $ArchiveRoot"
Write-Host "Open UI with:"
Write-Host "mlflow ui --backend-store-uri `"$ArchiveRoot`" --host 127.0.0.1 --port 5001"
