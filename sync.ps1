# Push the latest code to GitHub in one step.
# Usage:  ./sync.ps1  "optional commit message"
param([string]$msg = "Update: work in progress")

git add -A
# Only commit if there are staged changes.
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m $msg
    git push
    Write-Host "Synced to GitHub." -ForegroundColor Green
} else {
    Write-Host "Nothing to sync (no code changes)." -ForegroundColor Yellow
}
