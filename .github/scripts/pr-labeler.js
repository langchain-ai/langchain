// Shared helpers for pr_labeler.yml and tag-external-issues.yml.
//
// Usage from actions/github-script (requires actions/checkout first):
//   const { h } = require('./.github/scripts/pr-labeler.js').loadAndInit(github, owner, repo, core);

const fs = require('fs');
const path = require('path');

function loadConfig() {
  const configPath = path.join(__dirname, 'pr-labeler-config.json');
  let raw;
  try {
    raw = fs.readFileSync(configPath, 'utf8');
  } catch (e) {
    throw new Error(`Failed to read ${configPath}: ${e.message}`);
  }
  let config;
  try {
    config = JSON.parse(raw);
  } catch (e) {
    throw new Error(`Failed to parse pr-labeler-config.json: ${e.message}`);
  }
  const required = [
    'labelColor', 'sizeThresholds', 'fileRules',
    'typeToLabel', 'scopeToLabel', 'trustedThreshold',
    'excludedFiles', 'excludedPaths',
  ];
  const missing = required.filter(k => !(k in config));
  if (missing.length > 0) {
    throw new Error(`pr-labeler-config.json missing required keys: ${missing.join(', ')}`);
  }
  return config;
}

function init(github, owner, repo, config, core) {
  if (!core) {
    throw new Error('init() requires a `core` parameter (e.g., from actions/github-script)');
  }
  const {
    trustedThreshold,
    labelColor,
    sizeThresholds,
    scopeToLabel,
    typeToLabel,
    fileRules: fileRulesDef,
    excludedFiles,
    excludedPaths,
  } = config;

  const sizeLabels = sizeThresholds.map(t => t.label);
  const allTypeLabels = [...new Set(Object.values(typeToLabel))];
  const tierLabels = ['new-contributor', 'trusted-contributor'];

  // ── Label management ──────────────────────────────────────────────

  async function ensureLabel(name, color = labelColor) {
    try {
      await github.rest.issues.getLabel({ owner, repo, name });
    } catch (e) {
      if (e.status !== 404) throw e;
      try {
        await github.rest.issues.createLabel({ owner, repo, name, color });
      } catch (createErr) {
        // 422 = label created by a concurrent run between our get and create
        if (createErr.status !== 422) throw createErr;
        core.info(`Label "${name}" creation returned 422 (likely already exists)`);
      }
    }
  }

  // ── Size calculation ──────────────────────────────────────────────

  function getSizeLabel(totalChanged) {
    for (const t of sizeThresholds) {
      if (t.max != null && totalChanged < t.max) return t.label;
    }
    // Last entry has no max — it's the catch-all
    return sizeThresholds[sizeThresholds.length - 1].label;
  }

  function computeSize(files) {
    const excluded = new Set(excludedFiles);
    const totalChanged = files.reduce((sum, f) => {
      const p = f.filename ?? '';
      const base = p.split('/').pop();
      if (excluded.has(base)) return sum;
      for (const prefix of excludedPaths) {
        if (p.startsWith(prefix)) return sum;
      }
      return sum + (f.additions ?? 0) + (f.deletions ?? 0);
    }, 0);
    return { totalChanged, sizeLabel: getSizeLabel(totalChanged) };
  }

  // ── File-based labels ─────────────────────────────────────────────

  function buildFileRules() {
    return fileRulesDef.map((rule, i) => {
      let test;
      if (rule.prefix) test = p => p.startsWith(rule.prefix);
      else if (rule.suffix) test = p => p.endsWith(rule.suffix);
      else if (rule.exact) test = p => p === rule.exact;
      else if (rule.pattern) {
        const re = new RegExp(rule.pattern);
        test = p => re.test(p);
      } else {
        throw new Error(
          `fileRules[${i}] (label: "${rule.label}") has no recognized matcher ` +
          `(expected one of: prefix, suffix, exact, pattern)`
        );
      }
      return { label: rule.label, test, skipExcluded: !!rule.skipExcludedFiles };
    });
  }

  function matchFileLabels(files, fileRules) {
    const rules = fileRules || buildFileRules();
    const excluded = new Set(excludedFiles);
    const labels = new Set();
    for (const rule of rules) {
      // skipExcluded: ignore files whose basename is in the top-level
      // "excludedFiles" list (e.g. uv.lock) so lockfile-only changes
      // don't trigger package labels.
      const candidates = rule.skipExcluded
        ? files.filter(f => !excluded.has((f.filename ?? '').split('/').pop()))
        : files;
      if (candidates.some(f => rule.test(f.filename ?? ''))) {
        labels.add(rule.label);
      }
    }
    return labels;
  }

  // ── Title-based labels ────────────────────────────────────────────

  function matchTitleLabels(title) {
    const labels = new Set();
    const m = (title ?? '').match(/^(\w+)(?:\(([^)]+)\))?(!)?:/);
    if (!m) return { labels, type: null, typeLabel: null, scopes: [], breaking: false };

    const type = m[1].toLowerCase();
    const scopeStr = m[2] ?? '';
    const breaking = !!m[3];

    const typeLabel = typeToLabel[type] || null;
    if (typeLabel) labels.add(typeLabel);
    if (breaking) labels.add('breaking');

    const scopes = scopeStr.split(',').map(s => s.trim()).filter(Boolean);
    for (const scope of scopes) {
      const sl = scopeToLabel[scope];
      if (sl) labels.add(sl);
    }

    return { labels, type, typeLabel, scopes, breaking };
  }

  // ── Org membership ────────────────────────────────────────────────

  async function checkMembership(author, userType) {
    if (userType === 'Bot') {
      console.log(`${author} is a Bot — treating as internal`);
      return { isExternal: false };
    }

    try {
      const membership = await github.rest.orgs.getMembershipForUser({
        org: 'langchain-ai',
        username: author,
      });
      const isExternal = membership.data.state !== 'active';
      console.log(
        isExternal
          ? `${author} has pending membership — treating as external`
          : `${author} is an active member of langchain-ai`,
      );
      return { isExternal };
    } catch (e) {
      if (e.status === 404) {
        console.log(`${author} is not a member of langchain-ai`);
        return { isExternal: true };
      }
      // Non-404 errors (rate limit, auth failure, server error) must not
      // silently default to external — rethrow to fail the step.
      throw new Error(
        `Membership check failed for ${author} (${e.status}): ${e.message}`,
      );
    }
  }

  // ── Contributor analysis ──────────────────────────────────────────

  async function getContributorInfo(contributorCache, author, userType) {
    if (contributorCache.has(author)) return contributorCache.get(author);

    const { isExternal } = await checkMembership(author, userType);

    let mergedCount = null;
    if (isExternal) {
      try {
        const result = await github.rest.search.issuesAndPullRequests({
          q: `repo:${owner}/${repo} is:pr is:merged author:"${author}"`,
          per_page: 1,
        });
        mergedCount = result?.data?.total_count ?? null;
      } catch (e) {
        if (e?.status !== 422) throw e;
        core.warning(`Search failed for ${author}; skipping tier.`);
      }
    }

    const info = { isExternal, mergedCount };
    contributorCache.set(author, info);
    return info;
  }

  // ── Tier label resolution ───────────────────────────────────────────

  async function applyTierLabel(issueNumber, author, { skipNewContributor = false } = {}) {
    let mergedCount;
    try {
      const result = await github.rest.search.issuesAndPullRequests({
        q: `repo:${owner}/${repo} is:pr is:merged author:"${author}"`,
        per_page: 1,
      });
      mergedCount = result?.data?.total_count;
    } catch (error) {
      if (error?.status !== 422) throw error;
      core.warning(`Search failed for ${author}; skipping tier label.`);
      return;
    }

    if (mergedCount == null) {
      core.warning(`Search response missing total_count for ${author}; skipping tier label.`);
      return;
    }

    let tierLabel = null;
    if (mergedCount >= trustedThreshold) tierLabel = 'trusted-contributor';
    else if (mergedCount === 0 && !skipNewContributor) tierLabel = 'new-contributor';

    if (tierLabel) {
      await ensureLabel(tierLabel);
      await github.rest.issues.addLabels({
        owner, repo, issue_number: issueNumber, labels: [tierLabel],
      });
      console.log(`Applied '${tierLabel}' to #${issueNumber} (${mergedCount} merged PRs)`);
    } else {
      console.log(`No tier label for ${author} (${mergedCount} merged PRs)`);
    }

    return tierLabel;
  }

  return {
    ensureLabel,
    getSizeLabel,
    computeSize,
    buildFileRules,
    matchFileLabels,
    matchTitleLabels,
    allTypeLabels,
    checkMembership,
    getContributorInfo,
    applyTierLabel,
    sizeLabels,
    tierLabels,
    trustedThreshold,
    labelColor,
  };
}

function loadAndInit(github, owner, repo, core) {
  const config = loadConfig();
  return { config, h: init(github, owner, repo, config, core) };
}

module.exports = { loadConfig, init, loadAndInit };
