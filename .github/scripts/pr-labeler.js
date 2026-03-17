// Shared helpers for pr_labeler.yml and tag-external-issues.yml.
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
        if (createErr.status !== 422) throw createErr;
        // 422 = label created by a concurrent run between our get and create
        const core = require('@actions/core');
        core.info(`Label "${name}" creation returned 422 (likely already exists)`);
      }
    }
  }

  function getSizeLabel(totalChanged) {
    for (const t of config.sizeThresholds) {
      if (t.max != null && totalChanged < t.max) return t.label;
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
    checkMembership,
    getContributorInfo,
  };
}

module.exports = { loadConfig, init };
