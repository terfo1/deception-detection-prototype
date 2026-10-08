param(
    [Parameter(Mandatory = $true)][string]$DocumentPath,
    [Parameter(Mandatory = $true)][string]$PdfPath
)
$ErrorActionPreference = 'Stop'
$inputDocument = (Resolve-Path -LiteralPath $DocumentPath).Path
$outputPdf = [System.IO.Path]::GetFullPath($PdfPath)
if (Test-Path -LiteralPath $outputPdf) { throw "Output already exists: $outputPdf" }
$wordInstance = $null
$reportDocument = $null
try {
    # Separate hidden instance; never attach to or modify the user's open documents.
    $wordInstance = New-Object -ComObject Word.Application
    $wordInstance.Visible = $false
    $wordInstance.DisplayAlerts = 0
    $reportDocument = $wordInstance.Documents.Open($inputDocument, $false, $true)
    $reportDocument.Repaginate()
    $reportDocument.ExportAsFixedFormat($outputPdf, 17)
    Write-Output "Native Word PDF saved: $outputPdf; verify pagination in the exported PDF"
}
finally {
    if ($null -ne $reportDocument) {
        $reportDocument.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($reportDocument)
    }
    if ($null -ne $wordInstance) {
        $wordInstance.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordInstance)
    }
}
