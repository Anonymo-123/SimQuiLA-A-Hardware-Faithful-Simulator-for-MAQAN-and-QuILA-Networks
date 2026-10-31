# How we work

## First time
1. Accept the invitation e-mail from GitHub.
2. Follow "Get started" in the README (clone, install, run `pytest`).

## Each time you work on something
```bash
git switch main
git pull                          # get everyone's latest changes
git switch -c your-name/short-topic   # your own branch, e.g. ravi/detector-model
# ... edit code ...
pytest                            # make sure nothing broke
git add -A
git commit -m "Describe what you changed"
git push -u origin your-name/short-topic
```
Then open the repository on GitHub, click **Compare & pull request**, and
describe your change. Someone else looks at it, and it gets merged into `main`.

Prefer the mouse? **GitHub Desktop** (desktop.github.com) does all of the
above with buttons.

## A few rules
- Work on your own branch, not directly on `main`.
- Run `pytest` before you push.
- If you change any physics or key-rate calculation, add a test that checks
  it against a known formula or a published result.
- Don't change `sequence/` unless you have to; if you do, note it in `PATCHES.md`.
- **Never upload MAQAN measurement data, internal documents, passwords or keys.**
  Keep data in the `data/` folder on your own computer; Git ignores it.
