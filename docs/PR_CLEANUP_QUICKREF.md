# Quick Reference: Duplicate PR Cleanup

## TL;DR

```bash
# Close duplicate PRs automatically
./scripts/close_duplicate_prs.sh
```

## What Gets Closed

| PR # | Reason |
|------|--------|
| #17  | Duplicate of merged PR #2 |
| #28  | Duplicate of merged PR #31 |

## Manual Review Needed

- **PR #12** - Check for unique changes vs merged PR #2/#8
- **PR #32** - Compare with PR #9 and consolidate

## Files

- 📊 **Analysis**: `docs/PR_CLEANUP_REPORT.md`
- 📋 **Summary**: `docs/PR_CLEANUP_SUMMARY.md`
- 🔧 **Tools**: `scripts/close_duplicate_prs.sh`, `scripts/cleanup_duplicate_prs.py`
- 📖 **Docs**: `scripts/README.md`

## Commands

```bash
# Option 1: Use the script
./scripts/close_duplicate_prs.sh

# Option 2: Manual closure
gh pr close 17 --comment "Duplicate of merged PR #2"
gh pr close 28 --comment "Duplicate of merged PR #31"

# Option 3: Check for new duplicates
python scripts/cleanup_duplicate_prs.py
```

## Need Help?

1. Read `docs/PR_CLEANUP_SUMMARY.md` for full guide
2. Check `scripts/README.md` for tool documentation
3. Review `docs/PR_CLEANUP_REPORT.md` for detailed analysis
