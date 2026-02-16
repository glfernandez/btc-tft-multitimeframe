# Security Checklist (Public Push)

Run this checklist before pushing to a public branch.

## 1. Secrets
- Verify no API keys, access tokens, or private keys are in tracked files.
- Verify no `.env` files are tracked.

## 2. Privacy
- Verify no personal absolute local paths are present (for example `/Users/...`).
- Verify no personal email addresses are present in tracked files.

## 3. Data Scope
- Verify raw datasets are not tracked.
- Verify heavy logs/results are not tracked.
- Verify model checkpoints are only tracked if intentionally publishing reproducibility artifacts.

## 4. Git Hygiene
- Check tracked files:
  ```bash
  git ls-files
  ```
- Check unstaged/staged changes:
  ```bash
  git status
  ```

## 5. Optional Local Scan
```bash
rg -n --hidden -S "(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-|AIza[0-9A-Za-z\\-_]{35}|BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY|api[_-]?key\\s*[=:]|token\\s*[=:]|secret\\s*[=:])" .
rg -n --hidden -S "/Users/|gmail.com|@|GoogleDrive-" .
```
