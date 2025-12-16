#!/usr/bin/env bash
# Lightweight repo summary helper.
# Save as scripts/repo_summary.sh , make executable (chmod +x), then run ./scripts/repo_summary.sh

set -euo pipefail

echo "=== repo root: $(pwd) ==="
echo
echo "1) Git status / latest commit"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && {
  echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "last commit:"
  git --no-pager log -n 1 --pretty=format:"%h %ad | %an | %s" --date=iso
} || echo "Not a git repo"
echo
echo "2) Top-level files & directories (sizes)"
du -h --max-depth=1 2>/dev/null | sort -hr | sed 's/^\t//' | head -n 40
echo
echo "3) Quick tree (2 levels)"
if command -v tree >/dev/null 2>&1; then
  tree -L 2 -a
else
  find . -maxdepth 2 -mindepth 1 -printf "%P\n" | sed 's@^@/@' | sed 's@^/@  @'
fi
echo
echo "4) Look for main scripts / entry points"
grep -R --line-number -E "if __name__ == .__main__.|argparse.ArgumentParser|def main\(" -- scripts 2>/dev/null || true
echo
echo "5) Count tests / requirements"
ls -1 tests 2>/dev/null | wc -l || echo "no tests dir"
[ -f requirements.txt ] && echo "requirements.txt exists" || echo "no requirements.txt"
echo
echo "6) Big files (top 20)"
find . -type f -printf "%s %p\n" 2>/dev/null | sort -nr | head -n 20 | awk '{printf "%10s %s\n", $1, $2}'
echo
echo "7) TODO / FIXME / NOTES occurrences"
grep -R --line-number -E "TODO|FIXME|HACK|NOTE:" 2>/dev/null || true
echo
echo "8) Search for output/results and data directories"
ls -ld results data */results */data 2>/dev/null || true
echo
echo "=== end repo summary ==="