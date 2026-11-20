# Package releases

Packages in `release-please-config.json` are independently versioned. When changes land on `master`, release-please opens or refreshes one **draft** release PR per package with a version bump and `CHANGELOG.md`. The setup is ported from Deep Agents, including its pinned release-please runner, local Git history reader, and fixes for `==` tags and single-file exclusions.

## Normal releases

1. Merge Conventional Commits into `master`. Release scope comes from changed package paths, not the title scope. `feat`, `fix`, `perf`, and `revert` produce release notes; tests and `uv.lock` alone do not trigger releases.
2. Wait for the Release Please workflow and its lockfile job to finish. Do not edit release branches while automation is refreshing them.
3. Inspect the package's draft release PR, version files, generated changelog, and CI. Mark it ready yourself when it is ready to ship.
4. Merge the release PR after required checks pass. **Merging automatically dispatches the existing Package Release workflow and publishes to PyPI.** Watch that separate workflow for publication results.

Release PR scopes use PyPI names, such as `release(langchain-core): 1.8.5`. The tag remains `langchain-core==1.8.5`. Ordinary contribution scopes such as `fix(core): ...` remain unchanged.

Additive features and fixes produce patch releases, matching the repository's existing release cadence. Explicit breaking changes retain release-please's default major bump for stable packages and minor bump before 1.0. Use `Release-As: X.Y.Z` on a real package-scoped commit when intentionally choosing another version. Never use an empty commit: without changed paths, release-please cannot safely assign a package.

The manifest starts from existing published stable releases, not a fresh `0.1.0` baseline. Historical tags are not recreated, and this migration does not bump package versions.

## Publishing and safety

The automatic path dispatches `_release.yml` directly, rather than calling it as a reusable workflow, preserving the workflow identity configured for PyPI trusted publishing. Existing build, wheel checksums, dependency checks, unit and integration tests, downstream compatibility tests, TestPyPI, and PyPI gates remain in place.

`release-sha` pins every artifact and test checkout and the final GitHub tag to the release PR's merge commit. Label finalization reads the current automation on `master`. The workflow validates that this exact SHA is reachable from `master`, unless the existing non-master release override is explicitly enabled. A newer commit landing during dispatch does not change the package being published.

The `auto:release-pending` label stays on a merged release PR until publishing and GitHub tagging succeed. Only then does the publisher replace it with `auto:release-tagged`. Release PR maintenance defers while any merged pending release exists, avoiding version downgrades while tags are missing. Completion of a Package Release workflow retries maintenance automatically; ordinary pushes and manual dispatch also retry it. Failed API reads fail closed.

## One-time repository setup

Before enabling this workflow, confirm the GitHub App used by `ORG_MEMBERSHIP_APP_CLIENT_ID` and `ORG_MEMBERSHIP_APP_PRIVATE_KEY` is installed on this repository with **Contents**, **Issues**, and **Pull requests** read/write permissions. These secret names already serve the PR labeler; the release workflow requests repository-limited permissions. If that App is intentionally membership-only, configure a release-capable App in those secrets only after reviewing the labeler's needs, or adapt the new workflow to dedicated release App secrets.

Release PRs and lockfile commits use an App token so GitHub emits ordinary PR events and required CI runs. Do not replace it with `GITHUB_TOKEN` without also arranging CI dispatch: GitHub suppresses workflows triggered by that token's writes. Dependency installation runs with lifecycle scripts disabled and before the release token is minted.

The existing `Release` environment and PyPI/TestPyPI trusted-publisher registrations remain unchanged. No secrets, environments, repository permissions, labels, branch rules, or published artifacts are configured by the migration PR itself. Labels are created as needed by release-please.

## Manual releases and recovery

The existing Package Release workflow remains available for hotfixes and prereleases. Supply a package, expected `release-version`, and preferably an exact `release-sha`:

```bash
gh workflow run _release.yml --repo langchain-ai/langchain \
  -f working-directory=core \
  -f release-version=1.8.5 \
  -f release-sha=<full-commit-sha>
```

Dispatch publishes real artifacts; this is not a dry run. For releases outside `master`, retain the existing explicit `dangerous-nonmaster-release` override. Prerelease dependency resolution still requires the existing `allow-prereleases` opt-in.

If publishing fails before PyPI publication, fix the cause and redispatch the publisher with the same version and the intended source SHA. If that version is already on PyPI, do not rebuild or republish different code under it: ship a new version. If PyPI publication and tagging succeeded but label finalization failed, rerun only the label job, or manually replace the pending label after verifying the tag and release. Do not rerun the full publisher merely to fix labels.

If release PR updates stay deferred, inspect merged PRs labeled `auto:release-pending` and their Package Release runs. Recover the publish first; do not remove pending labels while a publish is still running or failed before tagging. Then manually dispatch Release Please to refresh remaining draft PRs.

## Adding packages and maintaining the runner

Add each package to both configuration and manifest, with its PyPI component name, Python version files, and exclusions for tests and lockfiles. For an already-published package, seed the manifest from its latest stable published `==` tag; for a never-published package, use `0.0.0` as the baseline. Add the component to PR title lint's accepted scopes and keep the publisher dropdown synchronized.

The isolated runner pins `release-please` in `release-please/package.json` and its pnpm lockfile. When upgrading, reapply the exact-file exclusion and `==` tag patches, retain the minimatch security override, run the focused tests, and audit the dependency tree. The custom Python updater and patch-version strategy require this runner, not the stock action or CLI.

From the repository root:

```bash
corepack pnpm@12.18.0 --dir .github/release-please install --frozen-lockfile --ignore-scripts
corepack pnpm@12.18.0 --dir .github/release-please run lint
node --test .github/scripts/release-please*.test.* .github/release-please/*.test.mjs
corepack pnpm@12.18.0 --dir .github/release-please audit
```

After the configuration has landed on `master`, `DRY_RUN=true GITHUB_REPOSITORY=langchain-ai/langchain node .github/release-please/run.mjs` previews PRs without writing them. Provide `GITHUB_TOKEN` securely in the environment; do not paste tokens into commands. Never dispatch the publisher as a test of this setup.
