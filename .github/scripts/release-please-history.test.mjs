import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { after, before, test } from 'node:test';

import { fetchOverrideBodies, gitMergeCommits } from '../release-please/history.mjs';

let repo;

function git(...args) {
  return execFileSync('git', args, { cwd: repo, encoding: 'utf8' }).trim();
}

function commitFiles(message, files) {
  for (const [path, content] of Object.entries(files)) {
    mkdirSync(dirname(join(repo, path)), { recursive: true });
    writeFileSync(join(repo, path), content);
  }
  git('add', '-A');
  git('commit', '-q', '-m', message);
  return git('rev-parse', 'HEAD');
}

async function collect(options = {}) {
  const commits = [];
  for await (const commit of gitMergeCommits('master', { targetBranch: 'master', cwd: repo, ...options })) {
    commits.push(commit);
  }
  return commits;
}

before(() => {
  repo = mkdtempSync(join(tmpdir(), 'release-please-history-'));
  git('init', '-q', '-b', 'master');
  git('config', 'user.email', 'test@example.com');
  git('config', 'user.name', 'Test');
  git('config', 'commit.gpgsign', 'false');

  commitFiles('chore: initial', { 'libs/a/a.py': 'a\n', 'libs/b/b.py': 'b\n' });
  commitFiles('feat(a): add thing ✨ (#10)\n\nBody of the PR.', { 'libs/a/a.py': 'a2\n' });

  git('switch', '-q', '-c', 'feature');
  commitFiles('feat(a)!: break a', { 'libs/a/a.py': 'a3\n' });
  commitFiles('chore(deps): relock b', { 'libs/b/uv.lock': 'lock\n' });
  git('switch', '-q', 'master');
  git('merge', '-q', '--no-ff', 'feature', '-m', 'Merge pull request #11 from org/feature\n\nfeat(a)!: break a');

  renameSync(join(repo, 'libs/b/b.py'), join(repo, 'libs/b/c.py'));
  commitFiles('refactor(b): rename module (#12)', {});
});

after(() => rmSync(repo, { recursive: true, force: true }));

test('yields squash commits with their PR, newest first', async () => {
  const commits = await collect();
  assert.equal(commits[0].message, 'refactor(b): rename module (#12)');
  const feat = commits.find(c => c.pullRequest?.number === 10);
  assert.equal(feat.pullRequest.title, 'feat(a): add thing ✨');
  assert.equal(feat.message, 'feat(a): add thing ✨ (#10)\n\nBody of the PR.');
  assert.equal(feat.pullRequest.body, feat.message);
  assert.deepEqual(feat.files, ['libs/a/a.py']);
  assert.equal(commits.at(-1).message, 'chore: initial');
  assert.equal(commits.at(-1).pullRequest, undefined);
});

test('lists a merge commit\'s files against its first parent and keeps branch commits', async () => {
  const commits = await collect();
  const merge = commits.find(c => c.pullRequest?.number === 11);
  assert.deepEqual(merge.files.sort(), ['libs/a/a.py', 'libs/b/uv.lock']);
  assert.equal(merge.pullRequest.headBranchName, 'org/feature');
  assert.ok(commits.some(c => c.message === 'feat(a)!: break a'));
  assert.ok(commits.some(c => c.message === 'chore(deps): relock b'));
});

test('lists a rename as its old and new paths', async () => {
  const [rename] = await collect();
  assert.deepEqual(rename.files.sort(), ['libs/b/b.py', 'libs/b/c.py']);
});

test('uses the PR body when it carries a commit override', async () => {
  const body = 'Text\n\nBEGIN_COMMIT_OVERRIDE\nfix(a): corrected (#10)\nEND_COMMIT_OVERRIDE';
  const commits = await collect({ overrideBodies: new Map([[10, body]]) });
  const feat = commits.find(c => c.pullRequest?.number === 10);
  assert.equal(feat.pullRequest.body, body);
  assert.match(feat.message, /^feat\(a\)/);
});

test('honors maxResults and stops git when the caller stops reading', async () => {
  assert.equal((await collect({ maxResults: 2 })).length, 2);
  for await (const commit of gitMergeCommits('master', { targetBranch: 'master', cwd: repo })) {
    assert.ok(commit.sha);
    break;
  }
});

test('throws when git log fails', async () => {
  await assert.rejects(async () => {
    for await (const commit of gitMergeCommits('no-such-ref', { targetBranch: 'master', cwd: repo })) {
      assert.fail(`unexpected commit ${commit.sha}`);
    }
  }, /git log exited with code/);
});

test('fetchOverrideBodies keeps only bodies with an override and pages until a short page', async () => {
  const pages = [
    Array.from({ length: 100 }, (_, i) => ({ number: i + 1, body: i === 0 ? 'BEGIN_COMMIT_OVERRIDE x' : 'mentions it elsewhere' })),
    [{ number: 500, body: 'BEGIN_COMMIT_OVERRIDE y' }, { number: 501, body: null }],
  ];
  const urls = [];
  const fetchImpl = async url => {
    urls.push(url);
    return { ok: true, json: async () => ({ items: pages[urls.length - 1] }) };
  };
  const bodies = await fetchOverrideBodies({ owner: 'o', repo: 'r', token: 't', fetchImpl });
  assert.deepEqual([...bodies.keys()], [1, 500]);
  assert.equal(urls.length, 2);
  assert.match(decodeURIComponent(urls[0]), /repo:o\/r is:pr is:merged "BEGIN_COMMIT_OVERRIDE" in:body/);
});

test('fetchOverrideBodies fails on an API error', async () => {
  const fetchImpl = async () => ({ ok: false, status: 403 });
  await assert.rejects(fetchOverrideBodies({ owner: 'o', repo: 'r', token: 't', fetchImpl }), /HTTP 403/);
});


test('fetchOverrideBodies refuses incomplete results rather than silently losing overrides', async () => {
  for (const result of [{ incomplete_results: true }, { total_count: 1001 }]) {
    const fetchImpl = async () => ({ ok: true, json: async () => ({ items: [], ...result }) });
    await assert.rejects(fetchOverrideBodies({ owner: 'o', repo: 'r', token: 't', fetchImpl }), /incomplete/);
  }
});

test('fetchOverrideBodies fails closed when all ten search pages are full', async () => {
  const fetchImpl = async () => ({ ok: true, json: async () => ({ items: Array(100).fill({ number: 1, body: '' }) }) });
  await assert.rejects(fetchOverrideBodies({ owner: 'o', repo: 'r', token: 't', fetchImpl }), /pagination limit/);
});
