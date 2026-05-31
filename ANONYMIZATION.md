# Anonymous submission checklist

Last audited: 2026-05-31

## Content audit (tracked files)

| Check | Status |
|-------|--------|
| Paper uses `\author{Anonymous Submission}` + `acl` review mode | OK |
| README omits author/affiliation | OK |
| No `.env` / API keys in tree (`.env` gitignored) | OK |
| No `sk-or-v1-*` secrets in tracked files | OK |
| No `/Users/...` home paths in tracked files | OK |
| No author name/email in workbench or derived `.md` | OK |

## What still identifies you (GitHub metadata)

| Leak | Where | Mitigation |
|------|-------|------------|
| GitHub username in remote URL | `Adya6714/retrieval-vs-computation` | Use [Anonymous GitHub](https://anonymous.4open.science/) on the branch below, or create a throwaway GitHub account + neutral repo name |
| Old commits on `master` | Author `Adya <srivastavadya@gmail.com>` | **Use branch `anonymous-submission`** (orphan, single commit, anonymous author) for the web service |
| This machine’s `git config` | Future local commits | Set author only for submission: see below |

## Branch for double-blind tools

```text
https://github.com/Adya6714/retrieval-vs-computation/tree/anonymous-submission
```

Paste that URL into [anonymous.4open.science](https://anonymous.4open.science/) (or your institution’s equivalent) to generate a reviewer-safe link.

## Local commits without your name

```bash
export GIT_AUTHOR_NAME="Anonymous"
export GIT_AUTHOR_EMAIL="anonymous@submission.invalid"
export GIT_COMMITTER_NAME="Anonymous"
export GIT_COMMITTER_EMAIL="anonymous@submission.invalid"
```

## Do not commit

- `.env`, API keys, `logs/api_runs/` (gitignored)
- Personal notes with name/affiliation
