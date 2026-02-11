#!/usr/bin/env python3
"""
Script to identify and document duplicate pull requests that need cleanup.

This script analyzes all pull requests in the repository and identifies:
1. Duplicate PRs where one has been merged (should be closed)
2. Multiple open incomplete PRs (should be consolidated/merged)

Usage:
    python scripts/cleanup_duplicate_prs.py [--output OUTPUT_FILE]
"""

import json
import argparse
import subprocess
import sys
from collections import defaultdict
from typing import List, Dict, Any, Tuple


def fetch_prs() -> List[Dict[str, Any]]:
    """Fetch all pull requests using GitHub CLI."""
    try:
        result = subprocess.run(
            ['gh', 'pr', 'list', '--state', 'all', '--limit', '1000', '--json',
             'number,title,state,mergeCommit,body,createdAt,headRefName'],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error fetching PRs: {e}", file=sys.stderr)
        print("Make sure GitHub CLI (gh) is installed and authenticated.", file=sys.stderr)
        sys.exit(1)


def normalize_title(title: str) -> str:
    """Normalize PR title for comparison."""
    normalized = title.lower().strip()
    # Remove common prefixes
    for prefix in ['[wip]', '🔒', '🎨', '🧪', '🧹', '[security fix description]']:
        normalized = normalized.replace(prefix, '').strip()
    return normalized


def analyze_duplicates(prs: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict]], List[Tuple]]:
    """
    Analyze PRs to find duplicates.
    
    Returns:
        Tuple of (duplicate_groups, actions_to_take)
    """
    # Group by normalized titles
    title_groups = defaultdict(list)
    for pr in prs:
        title = normalize_title(pr['title'])
        title_groups[title].append(pr)
    
    # Find duplicates (groups with more than one PR)
    duplicates = {t: g for t, g in title_groups.items() if len(g) > 1}
    
    # Determine actions
    actions = []
    
    for title, group in duplicates.items():
        # Sort by number (oldest first)
        group.sort(key=lambda x: x['number'])
        
        merged_prs = [pr for pr in group if pr.get('mergeCommit')]
        closed_prs = [pr for pr in group if pr['state'] == 'CLOSED' and not pr.get('mergeCommit')]
        open_prs = [pr for pr in group if pr['state'] == 'OPEN']
        
        if merged_prs and open_prs:
            # Close open duplicates that have a merged twin
            for pr in open_prs:
                actions.append((
                    'close',
                    pr['number'],
                    f"Duplicate of merged PR #{merged_prs[0]['number']}",
                    pr['title']
                ))
        elif open_prs and len(open_prs) > 1:
            # Multiple open incomplete PRs - recommend consolidation
            keep_pr = open_prs[0]  # Keep the oldest
            for pr in open_prs[1:]:
                actions.append((
                    'close',
                    pr['number'],
                    f"Duplicate of PR #{keep_pr['number']}, consolidate changes there",
                    pr['title']
                ))
    
    return duplicates, actions


def find_related_prs(prs: List[Dict[str, Any]]) -> List[Tuple]:
    """Find PRs that appear to be related/duplicates based on naming patterns."""
    related_actions = []
    
    # Find PRs with similar base names
    pr_map = {pr['number']: pr for pr in prs}
    
    for pr in prs:
        if pr['state'] != 'OPEN':
            continue
        
        title_lower = pr['title'].lower()
        
        # Check for explicit "sub pr" or "again" references
        if 'sub pr' in title_lower or ' again' in title_lower:
            # Try to extract referenced PR number
            import re
            matches = re.findall(r'pr[#\s]*(\d+)', title_lower)
            if matches:
                ref_number = int(matches[0])
                if ref_number in pr_map:
                    related_actions.append((
                        'close',
                        pr['number'],
                        f"Retry/duplicate of PR #{ref_number}",
                        pr['title']
                    ))
    
    return related_actions


def generate_report(duplicates: Dict, actions: List[Tuple], related_actions: List[Tuple]) -> str:
    """Generate a markdown report of duplicate PRs and recommended actions."""
    report = []
    report.append("# Duplicate Pull Requests Cleanup Report\n")
    report.append(f"Generated: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}\n")
    report.append("---\n\n")
    
    report.append("## Summary\n\n")
    report.append(f"- **Duplicate groups found**: {len(duplicates)}\n")
    report.append(f"- **Total PRs in duplicate groups**: {sum(len(g) for g in duplicates.values())}\n")
    report.append(f"- **PRs to close**: {len(actions) + len(related_actions)}\n\n")
    
    report.append("## Duplicate Groups\n\n")
    
    for title in sorted(duplicates.keys()):
        group = duplicates[title]
        group.sort(key=lambda x: x['number'])
        
        report.append(f"### Group: {title[:80]}\n\n")
        
        merged_prs = [pr for pr in group if pr.get('mergeCommit')]
        open_prs = [pr for pr in group if pr['state'] == 'OPEN']
        
        for pr in group:
            has_merge = pr.get('mergeCommit') is not None
            state = pr['state']
            
            if has_merge:
                status = "✅ MERGED"
            elif state == 'CLOSED':
                status = "❌ CLOSED"
            else:
                status = "🔵 OPEN"
            
            report.append(f"- **PR #{pr['number']}** - {status}\n")
            report.append(f"  - Title: {pr['title']}\n")
            report.append(f"  - Created: {pr['createdAt']}\n")
        
        if merged_prs and open_prs:
            report.append(f"\n  **⚠️ Action Required**: Close open PRs {[pr['number'] for pr in open_prs]} (merged twin: #{merged_prs[0]['number']})\n")
        elif open_prs and len(open_prs) > 1:
            report.append(f"\n  **⚠️ Action Required**: Consolidate {[pr['number'] for pr in open_prs]}\n")
        
        report.append("\n")
    
    if related_actions:
        report.append("## Related/Retry PRs\n\n")
        for action_type, pr_num, reason, title in related_actions:
            report.append(f"- **PR #{pr_num}**: {title}\n")
            report.append(f"  - **Action**: CLOSE - {reason}\n\n")
    
    report.append("## Actions to Take\n\n")
    report.append("### PRs to Close\n\n")
    
    all_actions = actions + related_actions
    if all_actions:
        report.append("```bash\n")
        for action_type, pr_num, reason, title in sorted(all_actions, key=lambda x: x[1]):
            report.append(f"# Close PR #{pr_num}: {reason}\n")
            report.append(f'gh pr close {pr_num} --comment "Closing as {reason}"\n\n')
        report.append("```\n\n")
    else:
        report.append("No actions required - all duplicates already resolved! ✅\n\n")
    
    report.append("## Manual Review Required\n\n")
    report.append("Please manually review the following to ensure accuracy:\n\n")
    report.append("1. Verify that closed PRs truly duplicate merged PRs\n")
    report.append("2. Check if any open duplicate PRs contain unique changes that should be preserved\n")
    report.append("3. Ensure PR descriptions and comments are preserved/migrated as needed\n\n")
    
    return ''.join(report)


def main():
    parser = argparse.ArgumentParser(
        description='Identify and document duplicate pull requests'
    )
    parser.add_argument(
        '--output', '-o',
        default='docs/PR_CLEANUP_REPORT.md',
        help='Output file for the report (default: docs/PR_CLEANUP_REPORT.md)'
    )
    args = parser.parse_args()
    
    print("Fetching pull requests...")
    prs = fetch_prs()
    print(f"Found {len(prs)} total pull requests")
    
    print("Analyzing for duplicates...")
    duplicates, actions = analyze_duplicates(prs)
    related_actions = find_related_prs(prs)
    
    print(f"Found {len(duplicates)} duplicate groups")
    print(f"Recommended actions: {len(actions) + len(related_actions)} PRs to close")
    
    print(f"\nGenerating report to {args.output}...")
    report = generate_report(duplicates, actions, related_actions)
    
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {args.output}")
    print("\nPreview of actions:")
    for action_type, pr_num, reason, title in actions + related_actions:
        print(f"  - Close PR #{pr_num}: {reason}")


if __name__ == '__main__':
    main()
