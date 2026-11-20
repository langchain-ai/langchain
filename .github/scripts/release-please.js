const { execFileSync } = require('node:child_process');
const { readFileSync } = require('node:fs');

const PREFIX = 'release-please--branches--master--components--';
const PENDING = 'auto:release-pending';
const TAGGED = 'auto:release-tagged';

function git(...args) {
  return execFileSync('git', args, { encoding: 'utf8', timeout: 60_000 }).trim();
}

function packages() {
  const config = JSON.parse(readFileSync('release-please-config.json', 'utf8'));
  for (const path of Object.keys(config.packages)) {
    if (!/^libs\/(?:partners\/)?[a-z0-9_-]+$/.test(path)) {
      throw new Error(`Invalid managed package path: ${path}`);
    }
  }
  return config.packages;
}

function releaseTarget(pr, managed) {
  if (!pr.merged_at || pr.base.ref !== 'master' || !pr.head.ref.startsWith(PREFIX)) return;
  const component = pr.head.ref.slice(PREFIX.length);
  const entry = Object.entries(managed).find(([, value]) => value.component === component);
  if (!entry) throw new Error(`Unknown release component: ${component}`);
  const [path, config] = entry;
  const match = pr.title.match(/^release\(([^)]+)\): (\d+\.\d+\.\d+)$/);
  if (!match || match[1] !== component) throw new Error(`Invalid release title: ${pr.title}`);
  return { path, component, version: match[2], changelog: `${path}/${config['changelog-path'] || 'CHANGELOG.md'}`, number: pr.number };
}

async function pendingReleases(github, repo) {
  const issues = await github.paginate(github.rest.issues.listForRepo, {
    ...repo, state: 'closed', labels: PENDING, per_page: 100,
  });
  const pending = [];
  for (const issue of issues) {
    if (!issue.pull_request) continue;
    const { data: pr } = await github.rest.pulls.get({ ...repo, pull_number: issue.number });
    if (pr.merged_at) pending.push(pr);
  }
  return pending;
}

async function dispatchRelease({ github, context, core }) {
  core.setOutput('dispatched', 'false');
  if (context.eventName !== 'push') return;
  const changed = git('diff-tree', '--root', '--no-commit-id', '--name-only', '-r', '--diff-merges=first-parent', context.sha)
    .split('\n').filter(Boolean);
  if (!changed.length) throw new Error('Empty commits cannot be scoped safely by release-please.');
  const prs = await github.paginate(github.rest.repos.listPullRequestsAssociatedWithCommit, {
    ...context.repo, commit_sha: context.sha, per_page: 100,
  });
  const managed = packages();
  const candidates = prs.filter(pr => pr.merge_commit_sha === context.sha &&
    pr.labels.some(label => label.name === PENDING));
  const targets = candidates.map(pr => releaseTarget(pr, managed)).filter(Boolean);
  if (targets.length > 1) throw new Error('Expected at most one merged release PR per commit.');
  for (const target of targets) {
    if (!changed.includes(target.changelog) || !changed.includes('.release-please-manifest.json')) {
      throw new Error('Release commit must update its changelog and release manifest.');
    }
    const manifest = JSON.parse(readFileSync('.release-please-manifest.json', 'utf8'));
    if (manifest[target.path] !== target.version) throw new Error('Release title and manifest version differ.');
    await github.rest.actions.createWorkflowDispatch({
      ...context.repo, workflow_id: '_release.yml', ref: 'master',
      inputs: {
        'working-directory': target.path.replace(/^libs\/(?:partners\/)?/, ''),
        'release-version': target.version,
        'release-sha': context.sha,
      },
    });
    core.setOutput('dispatched', 'true');
    await core.summary.addHeading('Package release dispatched')
      .addLink(`${target.component} ${target.version}`, `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/workflows/_release.yml`)
      .addRaw(`\n\nArtifact commit: \`${context.sha}\`. Publishing runs separately; dispatch is not proof of publication.\n`)
      .write();
  }
}

async function guardPending({ github, context, core }) {
  const pending = await pendingReleases(github, context.repo);
  core.setOutput('skip', pending.length ? 'true' : 'false');
  if (pending.length) {
    const names = pending.map(pr => `#${pr.number} ${pr.title}`).join('; ');
    core.warning(`Release PR updates deferred until pending publishes finish: ${names}`);
    await core.summary.addHeading('Release PR updates deferred')
      .addRaw(`Merged release PRs still awaiting a tag: ${names}. A completed package-release workflow retries maintenance automatically. Recover failed publishes before clearing pending labels.\n`)
      .write();
  }
}

async function finalizeRelease({ github, context }) {
  const managed = packages();
  const { PACKAGE_NAME: component, RELEASE_VERSION: version } = process.env;
  const pending = await pendingReleases(github, context.repo);
  const matches = pending.filter(pr => {
    const target = releaseTarget(pr, managed);
    return target?.component === component && target.version === version;
  });
  if (matches.length > 1) throw new Error('Multiple merged pending PRs match this release.');
  for (const pr of matches) {
    await github.rest.issues.addLabels({ ...context.repo, issue_number: pr.number, labels: [TAGGED] });
    await github.rest.issues.removeLabel({ ...context.repo, issue_number: pr.number, name: PENDING });
  }
}

module.exports = { dispatchRelease, finalizeRelease, guardPending, pendingReleases, releaseTarget };
