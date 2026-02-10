#!/bin/bash
# Script to close duplicate pull requests
# Run this after reviewing docs/PR_CLEANUP_REPORT.md

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Duplicate PR Cleanup Script ===${NC}"
echo ""

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: GitHub CLI (gh) is not installed${NC}"
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo -e "${YELLOW}Not authenticated with GitHub CLI${NC}"
    echo "Running: gh auth login"
    gh auth login
fi

echo -e "${YELLOW}This script will close the following PRs:${NC}"
echo ""
echo "  - PR #17: Duplicate of merged PR #2"
echo "  - PR #28: Duplicate of merged PR #31"
echo ""
echo -e "${YELLOW}PRs requiring manual review (not closed by this script):${NC}"
echo "  - PR #12: Related to merged work"
echo "  - PR #32: Retry of PR #9"
echo ""

read -p "Do you want to proceed? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${GREEN}Closing duplicate PRs...${NC}"
echo ""

# Close PR #17
echo -e "${YELLOW}Closing PR #17...${NC}"
if gh pr close 17 --comment "Closing as duplicate of merged PR #2 (Jules/hive zero impl 6522049897564722866). All changes from this PR were already merged." 2>&1; then
    echo -e "${GREEN}✓ Closed PR #17${NC}"
else
    echo -e "${RED}✗ Failed to close PR #17 (may already be closed)${NC}"
fi
echo ""

# Close PR #28
echo -e "${YELLOW}Closing PR #28...${NC}"
if gh pr close 28 --comment "Closing as duplicate of merged PR #31. The fixes in this PR (device mismatch, gating bypass, shape errors, numerical stability) were already merged in PR #31." 2>&1; then
    echo -e "${GREEN}✓ Closed PR #28${NC}"
else
    echo -e "${RED}✗ Failed to close PR #28 (may already be closed)${NC}"
fi
echo ""

echo -e "${GREEN}=== Cleanup Complete ===${NC}"
echo ""
echo -e "${YELLOW}Manual review needed for:${NC}"
echo "  - PR #12: Run 'gh pr view 12' to review"
echo "  - PR #32: Run 'gh pr view 32' and compare with PR #9"
echo ""
echo "To close them after review:"
echo "  gh pr close 12 --comment \"Your reason\""
echo "  gh pr close 32 --comment \"Your reason\""
echo ""
echo "See docs/PR_CLEANUP_REPORT.md for full details."
