# Build paper/main.tex -> paper/main.pdf (run after every .tex change).
# MiKTeX pdflatex, two passes for references; packages auto-install.
$ErrorActionPreference = "Stop"
$paper = Join-Path $PSScriptRoot "..\paper"
Push-Location $paper
try {
    foreach ($pass in 1, 2) {
        pdflatex -interaction=nonstopmode -halt-on-error -enable-installer main.tex | Select-String -Pattern "^!|Output written|Warning" | Select-Object -First 20
        if ($LASTEXITCODE -ne 0) { throw "pdflatex failed on pass $pass (see paper/main.log)" }
    }
    Write-Output "OK: paper/main.pdf built"
} finally {
    Pop-Location
}
