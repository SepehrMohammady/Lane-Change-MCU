# Build paper/main.tex -> paper/main.pdf (run after every .tex change).
# MiKTeX pdflatex, two passes for references; packages auto-install.
# Output is captured before filtering: piping pdflatex straight into
# Select-Object -First stops the compiler once enough lines have matched.
$ErrorActionPreference = "Stop"
$paper = Join-Path $PSScriptRoot "..\paper"
Push-Location $paper
try {
    foreach ($pass in 1, 2) {
        $log = pdflatex -interaction=nonstopmode -halt-on-error -enable-installer main.tex
        $code = $LASTEXITCODE
        if ($pass -eq 2 -or $code -ne 0) {
            $log | Select-String -Pattern "^!|Output written|Warning" | Select-Object -First 20
        }
        if ($code -ne 0) { throw "pdflatex failed on pass $pass (see paper/main.log)" }
    }
    Write-Output "OK: paper/main.pdf built"
} finally {
    Pop-Location
}
