// Shared helpers for pr_labeler.yml (label + backfill jobs).
//
// Usage from actions/github-script:
//   const helpers = require('./.github/scripts/pr-labeler.js');
//   const config = helpers.loadConfig();
//   const { ensureLabel, getSizeLabel, ... } = helpers.init(github, owner, repo, config);

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

function init(github, owner, repo, config) {
  const { labelColor } = config;

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
      }
    }
  }

  function getSizeLabel(totalChanged) {
    for (const t of config.sizeThresholds) {
      if (t.max && totalChanged < t.max) return t.label;
    }
    // Last entry has no max — it's the catch-all (XL)
    return config.sizeThresholds[config.sizeThresholds.length - 1].label;
  }

  const sizeLabels = config.sizeThresholds.map(t => t.label);

  function computeSize(files) {
    const excluded = new Set(config.excludedFiles);
    const totalChanged = files.reduce((sum, f) => {
      const p = f.filename ?? '';
      const base = p.split('/').pop();
      if (config.excludedPaths.some(ep => p.startsWith(ep)) || excluded.has(base)) {
        return sum;
      }
      return sum + (f.additions ?? 0) + (f.deletions ?? 0);
    }, 0);
    return { totalChanged, sizeLabel: getSizeLabel(totalChanged) };
  }

  const depsPattern = /(?:^|\/)requirements[^/]*\.txt$/;

  function buildFileRules() {
    return config.fileRules.map(rule => {
      if (rule.pattern === 'deps') {
        return {
          label: rule.label,
          test: p => {
            const base = p.split('/').pop();
            return p.endsWith('pyproject.toml') ||
              base === 'uv.lock' ||
              depsPattern.test(p) ||
              p.endsWith('poetry.lock');
          },
        };
      }
      const prefixes = Array.isArray(rule.prefix) ? rule.prefix : [rule.prefix];
      return {
        label: rule.label,
        test: p => prefixes.some(pfx => p.startsWith(pfx)),
      };
    });
  }

  function matchFileLabels(files, fileRules) {
    const labels = new Set();
    for (const rule of fileRules) {
      if (files.some(f => rule.test(f.filename ?? ''))) {
        labels.add(rule.label);
      }
    }
    return labels;
  }

  function matchTitleLabels(title) {
    const labels = new Set();
    const titleMatch = title.match(/^(\w+)(?:\(([^)]+)\))?(!)?:/);
    if (!titleMatch) return { labels, type: null, scopes: [] };

    const type = titleMatch[1].toLowerCase();
    const scopeStr = titleMatch[2] ?? '';
    const breaking = !!titleMatch[3];

    const typeLabel = config.typeToLabel[type];
    if (typeLabel) labels.add(typeLabel);
    if (breaking) labels.add('breaking');

    const scopes = scopeStr.split(',').map(s => s.trim()).filter(Boolean);
    for (const scope of scopes) {
      const scopeLabel = config.scopeToLabel[scope];
      if (scopeLabel) labels.add(scopeLabel);
    }

    return { labels, type, scopes };
  }

  const allTypeLabels = [...new Set(Object.values(config.typeToLabel))];

  async function getContributorInfo(contributorCache, author, userType) {
    if (contributorCache.has(author)) return contributorCache.get(author);

    if (userType === 'Bot') {
      const info = { isExternal: false, mergedCount: 0 };
      contributorCache.set(author, info);
      return info;
    }

    let isExternal = true;
    try {
      const membership = await github.rest.orgs.getMembershipForUser({
        org: 'langchain-ai',
        username: author,
      });
      isExternal = membership.data.state !== 'active';
    } catch (e) {
      if (e.status !== 404) {
        const core = require('@actions/core');
        core.warning(`Membership check failed for ${author}: ${e.message}`);
      }
    }

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
        const core = require('@actions/core');
        core.warning(`Search failed for ${author}; skipping tier.`);
      }
    }

    const info = { isExternal, mergedCount };
    contributorCache.set(author, info);
    return info;
  }

  return {
    ensureLabel,
    getSizeLabel,
    sizeLabels,
    computeSize,
    buildFileRules,
    matchFileLabels,
    matchTitleLabels,
    allTypeLabels,
    getContributorInfo,
  };
}

module.exports = { loadConfig, init };
