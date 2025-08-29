#!/bin/bash
export GITHUB_TOKEN=""

# python fetch_issues.py --label rocm --limit 50 --dry-run --verbose > rocm_issues10.log 2>&1

python fetch_issues.py --limit 100 --dry-run --verbose > issues10.log 2>&1

# python fetch_issues.py --days 7 --label rocm --limit 30 --wait-for-action --verbose > rocm_issues_actual9.log 2>&1
# python fetch_issues.py --validate-source --days 30 --limit 10 --verbose