# Prépare MiniChat pour un autre PC Windows : dossier "publish" autonome (pas besoin d'installer .NET),
# puis une archive zip à copier.
#
#   powershell -ExecutionPolicy Bypass -File publish.ps1
#   powershell -ExecutionPolicy Bypass -File publish.ps1 -ModelsSource D:\MesModeles
#
# Le dossier publié contient : MiniChat.exe, le runtime .NET, TorchSharp et libtorch (CUDA + CPU),
# data/, et model/ + model-pretrain/ s'ils existent (par défaut dans bin\Debug\net8.0, là où "dotnet run" les crée).

param(
    [string]$Output = "publish",
    [string]$Zip = "MiniChat-win-x64.zip",
    [string]$ModelsSource = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Repart d'un dossier vide : pas de vieux fichiers d'une publication précédente.
if (Test-Path $Output) { Remove-Item -Recurse -Force $Output }

$publishArgs = @("publish", "-c", "Release", "-r", "win-x64", "--self-contained", "true", "-o", $Output)
if ($ModelsSource -ne "") { $publishArgs += "-p:ModelsSource=$((Resolve-Path $ModelsSource).Path)" }

dotnet @publishArgs
if ($LASTEXITCODE -ne 0) { throw "dotnet publish a échoué (code $LASTEXITCODE)." }

# ZipFile plutôt que Compress-Archive : Compress-Archive (PowerShell 5) échoue sur les fichiers de plus de 2 Go
# et est très lent ; libtorch CUDA fait plusieurs Go.
# Fichier par fichier plutôt que ZipFile.CreateFromDirectory : sous PowerShell 5, celui-ci écrit les chemins
# avec "\" ("data\texte.txt"), alors que le format zip impose "/". Les fichiers sont à la racine de l'archive.
Write-Host "Création de $Zip (quelques minutes : libtorch CUDA fait plusieurs Go)..."
$zipPath = Join-Path $PSScriptRoot $Zip
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$root = (Resolve-Path $Output).Path.TrimEnd('\')
$archive = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($file in Get-ChildItem $root -Recurse -File) {
        $entryName = $file.FullName.Substring($root.Length + 1).Replace('\', '/')
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive, $file.FullName, $entryName, [System.IO.Compression.CompressionLevel]::Optimal)
    }
}
finally { $archive.Dispose() }

$folderSize = (Get-ChildItem $Output -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
$zipSize = (Get-Item $zipPath).Length / 1GB
Write-Host ("Terminé : dossier {0} ({1:N1} Go), archive {2} ({3:N1} Go)." -f $Output, $folderSize, $Zip, $zipSize)
