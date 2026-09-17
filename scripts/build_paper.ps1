# Build paper/main.tex -> paper/main.pdf (run after every .tex change).
# MiKTeX pdflatex, packages auto-install. The TikZ diagrams in paper/figures/fig_*.tex
# are standalone documents compiled to PDF first and included as graphics, as the IEEE
# template expects; the data figures come from paper/figures/make_figures.py.
# Output is captured before filtering: piping pdflatex straight into
# Select-Object -First stops the compiler once enough lines have matched.
$ErrorActionPreference = "Stop"
$paper = Join-Path $PSScriptRoot "..\paper"
Push-Location (Join-Path $paper "figures")
try {
    foreach ($fig in Get-ChildItem -Filter "fig_*.tex") {
        $log = pdflatex -interaction=nonstopmode -halt-on-error -enable-installer $fig.Name
        if ($LASTEXITCODE -ne 0) {
            $log | Select-String -Pattern "^!" | Select-Object -First 5
            throw "pdflatex failed on figures/$($fig.Name)"
        }
    }
} finally {
    Pop-Location
}
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
