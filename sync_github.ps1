# ==========================================
# Enterprise GitHub Auto-Sync Pipeline
# ==========================================

Write-Host ">>> [1/4] Checking Git status..." -ForegroundColor Cyan
git status

Write-Host ">>> [2/4] Staging all modified files..." -ForegroundColor Cyan
git add .

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$commitMessage = "Auto-sync update: $timestamp"

Write-Host ">>> [3/4] Committing changes..." -ForegroundColor Cyan
git commit -m "$commitMessage"

Write-Host ">>> [4/4] Pushing changes to GitHub..." -ForegroundColor Cyan
git push origin main

Write-Host ">>> [SUCCESS] Repository successfully synchronized!" -ForegroundColor Green