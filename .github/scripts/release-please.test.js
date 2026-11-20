const assert = require('node:assert/strict');
const { after, before, test } = require('node:test');
const { execFileSync } = require('node:child_process');
const { mkdtempSync, mkdirSync, rmSync, writeFileSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const { dispatchRelease, finalizeRelease, guardPending, releaseTarget } = require('./release-please.js');

const managed = { 'libs/core': { component: 'langchain-core' } };
const release = {
  number: 42,
  title: 'release(langchain-core): 1.8.5',
  merged_at: '2026-11-20T00:00:00Z',
  base: { ref: 'master' },
  head: { ref: 'release-please--branches--master--components--langchain-core' },
  labels: [{ name: 'auto:release-pending' }],
};
const repo = { owner: 'langchain-ai', repo: 'langchain' };
const summary = {
  addHeading() { return this; },
  addLink() { return this; },
  addRaw() { return this; },
  async write() {},
};
function coreMock() {
  const outputs = {};
  return { outputs, setOutput(key, value) { outputs[key] = value; }, summary, warning() {} };
}
function githubMock(prs = []) {
  const calls = [];
  const github = {
    calls,
    async paginate(method) { return method(); },
    rest: {
      repos: { async listPullRequestsAssociatedWithCommit() { return prs; } },
      pulls: { async get({ pull_number }) { return { data: prs.find(pr => pr.number === pull_number) }; } },
      issues: {
        async listForRepo() { return prs.map(pr => ({ number: pr.number, pull_request: {} })); },
        async addLabels(args) { calls.push(['add', args]); },
        async removeLabel(args) { calls.push(['remove', args]); },
      },
      actions: { async createWorkflowDispatch(args) { calls.push(['dispatch', args]); } },
    },
  };
  return github;
}

let directory;
let originalCwd;
let sha;
function git(...args) {
  return execFileSync('git', args, { cwd: directory, encoding: 'utf8', timeout: 10_000 }).trim();
}
function commitFixture(message, ...parents) {
  const content = [
    `tree ${git('write-tree')}`,
    ...parents.map(parent => `parent ${parent}`),
    'author Release Fixture <fixture@example.com> 1700000000 +0000',
    'committer Release Fixture <fixture@example.com> 1700000000 +0000',
    '', message, '',
  ].join('\n');
  const commit = execFileSync('git', ['hash-object', '-t', 'commit', '-w', '--stdin'], {
    cwd: directory, input: content, encoding: 'utf8', timeout: 10_000,
  }).trim();
  git('update-ref', 'HEAD', commit);
  return commit;
}
before(() => {
  originalCwd = process.cwd();
  directory = mkdtempSync(join(tmpdir(), 'langchain-release-'));
  mkdirSync(join(directory, 'libs/core'), { recursive: true });
  writeFileSync(join(directory, 'release-please-config.json'), JSON.stringify({ packages: managed }));
  writeFileSync(join(directory, '.release-please-manifest.json'), JSON.stringify({ 'libs/core': '1.8.5' }));
  writeFileSync(join(directory, 'libs/core/CHANGELOG.md'), '# 1.8.5\n');
  git('init', '-q');
  git('add', '.');
  sha = commitFixture(release.title);
  process.chdir(directory);
});
after(() => {
  process.chdir(originalCwd);
  rmSync(directory, { recursive: true, force: true });
});

test('release target requires a merged master release branch and matching component', () => {
  assert.equal(releaseTarget({ ...release, merged_at: null }, managed), undefined);
  assert.equal(releaseTarget({ ...release, base: { ref: 'v1' } }, managed), undefined);
  assert.equal(releaseTarget({ ...release, head: { ref: 'manual-release' } }, managed), undefined);
  assert.equal(releaseTarget(release, managed).path, 'libs/core');
  assert.throws(() => releaseTarget({ ...release, title: 'release(openai): 1.8.5' }, managed), /Invalid release title/);
  assert.throws(() => releaseTarget(release, {}), /Unknown release component/);
});

test('merged release dispatch preserves the exact artifact commit and existing publisher', async () => {
  const github = githubMock([{ ...release, merge_commit_sha: sha }]);
  const core = coreMock();
  await dispatchRelease({ github, context: { repo, sha, eventName: 'push' }, core });
  assert.deepEqual(github.calls, [['dispatch', {
    ...repo, workflow_id: '_release.yml', ref: 'master',
    inputs: { 'working-directory': 'core', 'release-version': '1.8.5', 'release-sha': sha },
  }]]);
  assert.equal(core.outputs.dispatched, 'true');
});

test('manual version commits and unrelated associated PRs do not dispatch', async () => {
  for (const pr of [
    { ...release, labels: [], merge_commit_sha: sha },
    { ...release, merge_commit_sha: 'a'.repeat(40) },
    { ...release, head: { ref: 'manual-release' }, merge_commit_sha: sha },
  ]) {
    const github = githubMock([pr]);
    await dispatchRelease({ github, context: { repo, sha, eventName: 'push' }, core: coreMock() });
    assert.equal(github.calls.length, 0);
  }
});

test('workflow completion refreshes PRs without dispatching another release', async () => {
  const github = githubMock([release]);
  await dispatchRelease({ github, context: { repo, eventName: 'workflow_run' }, core: coreMock() });
  assert.equal(github.calls.length, 0);
});

test('mismatched manifest and unsafe package paths fail before dispatch', async () => {
  const github = githubMock([{ ...release, merge_commit_sha: sha, title: 'release(langchain-core): 1.8.6' }]);
  await assert.rejects(dispatchRelease({ github, context: { repo, sha, eventName: 'push' }, core: coreMock() }), /manifest version differ/);
  writeFileSync('release-please-config.json', JSON.stringify({ packages: { '../outside': {} } }));
  await assert.rejects(dispatchRelease({ github, context: { repo, sha, eventName: 'push' }, core: coreMock() }), /Invalid managed package path/);
  writeFileSync('release-please-config.json', JSON.stringify({ packages: managed }));
  assert.equal(github.calls.length, 0);
});

test('merge commits dispatch using their first-parent release changes', async () => {
  writeFileSync('.release-please-manifest.json', JSON.stringify({ 'libs/core': '1.8.4' }));
  writeFileSync('libs/core/CHANGELOG.md', '# 1.8.4\n');
  git('add', '.');
  const parent = commitFixture('chore(core): previous release', sha);
  writeFileSync('.release-please-manifest.json', JSON.stringify({ 'libs/core': '1.8.5' }));
  writeFileSync('libs/core/CHANGELOG.md', '# 1.8.5\n');
  git('add', '.');
  const merge = commitFixture(release.title, parent, sha);
  const github = githubMock([{ ...release, merge_commit_sha: merge }]);
  await dispatchRelease({ github, context: { repo, sha: merge, eventName: 'push' }, core: coreMock() });
  assert.equal(github.calls.length, 1);
  assert.equal(github.calls[0][1].inputs['release-sha'], merge);
});

test('empty and incomplete release commits cannot dispatch', async () => {
  writeFileSync('unrelated.txt', 'not a release\n');
  git('add', 'unrelated.txt');
  const incomplete = commitFixture(release.title, sha);
  const github = githubMock([{ ...release, merge_commit_sha: incomplete }]);
  await assert.rejects(dispatchRelease({ github, context: { repo, sha: incomplete, eventName: 'push' }, core: coreMock() }), /changelog and release manifest/);
  commitFixture('fix(core): empty', incomplete);
  await assert.rejects(dispatchRelease({ github, context: { repo, sha: git('rev-parse', 'HEAD'), eventName: 'push' }, core: coreMock() }), /Empty commits/);
  assert.equal(github.calls.length, 0);
});

test('pending publish defers maintenance; unmerged closed PRs do not', async () => {
  for (const [pr, expected] of [[release, 'true'], [{ ...release, merged_at: null }, 'false']]) {
    const core = coreMock();
    await guardPending({ github: githubMock([pr]), context: { repo }, core });
    assert.equal(core.outputs.skip, expected);
  }
  const github = githubMock();
  github.paginate = async () => { throw new Error('GitHub unavailable'); };
  const core = coreMock();
  await assert.rejects(guardPending({ github, context: { repo }, core }), /GitHub unavailable/);
  assert.equal(core.outputs.skip, undefined);
});

test('successful publication finalizes only the matching package/version', async () => {
  const oldName = process.env.PACKAGE_NAME;
  const oldVersion = process.env.RELEASE_VERSION;
  process.env.PACKAGE_NAME = 'langchain-core';
  process.env.RELEASE_VERSION = '1.8.5';
  try {
    const github = githubMock([release, { ...release, number: 43, title: 'release(langchain-core): 1.8.6' }]);
    await finalizeRelease({ github, context: { repo } });
    assert.deepEqual(github.calls.map(([method, args]) => [method, args.issue_number]), [['add', 42], ['remove', 42]]);
    const failed = githubMock([release]);
    failed.rest.issues.addLabels = async () => { throw new Error('label creation failed'); };
    await assert.rejects(finalizeRelease({ github: failed, context: { repo } }), /label creation failed/);
    assert.equal(failed.calls.length, 0);
  } finally {
    if (oldName === undefined) delete process.env.PACKAGE_NAME;
    else process.env.PACKAGE_NAME = oldName;
    if (oldVersion === undefined) delete process.env.RELEASE_VERSION;
    else process.env.RELEASE_VERSION = oldVersion;
  }
});
