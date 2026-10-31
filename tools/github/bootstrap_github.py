#!/usr/bin/env python3
"""Create the project's labels, milestones and issues on GitHub.

Reads ``tools/github/issues.json`` and uses the GitHub CLI (``gh``), which
must be installed (https://cli.github.com) and logged in (``gh auth login``).

Usage, from the repository root:

    python tools/github/bootstrap_github.py --repo YOUR-ORG/quila-sim --dry-run
    python tools/github/bootstrap_github.py --repo YOUR-ORG/quila-sim

Safe to re-run: labels are created or updated, and milestones and issues
whose titles already exist are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DATA_FILE = Path(__file__).with_name("issues.json")


def gh(args: list[str], dry_run: bool, stdin: str | None = None) -> str:
    """Run a gh command (or print it in dry-run mode) and return stdout."""
    if dry_run:
        print("  [dry-run] gh " + " ".join(args))
        return ""
    result = subprocess.run(
        ["gh", *args],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        sys.exit(f"gh {' '.join(args[:3])} ... failed:\n{result.stderr.strip()}")
    return result.stdout


def existing_titles(cmd: list[str], dry_run: bool) -> set[str]:
    out = gh(cmd, dry_run)
    return {line.strip() for line in out.splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="OWNER/REPO, e.g. my-org/quila-sim")
    parser.add_argument(
        "--dry-run", action="store_true", help="print actions without calling GitHub"
    )
    args = parser.parse_args()

    if not args.dry_run and shutil.which("gh") is None:
        sys.exit(
            "GitHub CLI 'gh' not found. Install it from https://cli.github.com "
            "and run 'gh auth login'."
        )

    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    repo = args.repo
    verb = "would create" if args.dry_run else "created"

    print(f"Labels ({len(data['labels'])})")
    for label in data["labels"]:
        gh(
            ["label", "create", label["name"], "--repo", repo, "--color", label["color"],
             "--description", label["description"], "--force"],
            args.dry_run,
        )

    print(f"Milestones ({len(data['milestones'])})")
    have = existing_titles(
        ["api", "--paginate", f"repos/{repo}/milestones?state=all", "--jq", ".[].title"],
        args.dry_run,
    )
    for ms in data["milestones"]:
        if ms["title"] in have:
            print(f"  exists: {ms['title']}")
            continue
        gh(
            ["api", f"repos/{repo}/milestones", "-f", f"title={ms['title']}",
             "-f", f"description={ms['description']}", "--silent"],
            args.dry_run,
        )
        print(f"  {verb}: {ms['title']}")

    print(f"Issues ({len(data['issues'])})")
    have = existing_titles(
        ["issue", "list", "--repo", repo, "--state", "all", "--limit", "1000",
         "--json", "title", "--jq", ".[].title"],
        args.dry_run,
    )
    for issue in data["issues"]:
        if issue["title"] in have:
            print(f"  exists: {issue['title']}")
            continue
        cmd = ["issue", "create", "--repo", repo, "--title", issue["title"], "--body-file", "-",
               "--milestone", issue["milestone"]]
        for label in issue["labels"]:
            cmd += ["--label", label]
        url = gh(cmd, args.dry_run, stdin=issue["body"]).strip()
        print(f"  {verb}: {issue['title']}  {url}".rstrip())

    print("Done.")


if __name__ == "__main__":
    main()
