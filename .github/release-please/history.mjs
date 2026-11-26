// Commit history for release-please from a local clone. run.mjs installs
// `gitMergeCommits` in place of release-please's GraphQL commit iterator.
// Kept free of release-please imports so it can be tested on its own.

import { spawn } from 'node:child_process';

const RECORD = '\x1e';
const MESSAGE_END = '\x1f';
const SQUASH_SUFFIX = /\s\(#(\d+)\)$/;
const MERGE_SUBJECT = /^Merge pull request #(\d+) from (.*)$/;

/**
 * Return the bodies of merged PRs that carry a `BEGIN_COMMIT_OVERRIDE` block,
 * keyed by PR number.
 *
 * release-please reads overrides from `commit.pullRequest.body`, and the
 * documented use is editing a PR body after merge, so the commit message alone
 * isn't enough. One search request finds every such PR.
 */
export async function fetchOverrideBodies({ owner, repo, token, fetchImpl = fetch }) {
  const bodies = new Map();
  const query = `repo:${owner}/${repo} is:pr is:merged "BEGIN_COMMIT_OVERRIDE" in:body`;
  for (let page = 1; page <= 10; page++) {
    const url = `https://api.github.com/search/issues?q=${encodeURIComponent(query)}&per_page=100&page=${page}`;
    const response = await fetchImpl(url, {
      signal: AbortSignal.timeout(30_000),
      headers: {
        Accept: 'application/vnd.github+json',
        Authorization: `Bearer ${token}`,
        'X-GitHub-Api-Version': '2022-11-28',
      },
    });
    if (!response.ok) {
      throw new Error(`Searching for BEGIN_COMMIT_OVERRIDE PRs failed: HTTP ${response.status}`);
    }
    const { items, incomplete_results: incomplete, total_count: total } = await response.json();
    if (incomplete || total > 1000) {
      throw new Error('Commit override search is incomplete; refusing to use partial release history');
    }
    for (const item of items) {
      if (item.body?.includes('BEGIN_COMMIT_OVERRIDE')) bodies.set(item.number, item.body);
    }
    if (items.length < 100) return bodies;
  }
  throw new Error('Commit override search reached its pagination limit');
}

function toCommit(record, targetBranch, overrideBodies) {
  const end = record.indexOf(MESSAGE_END);
  const head = record.slice(0, end);
  const newline = head.indexOf('\n');
  const sha = head.slice(0, newline);
  const message = head.slice(newline + 1).trim();
  const files = record.slice(end + 1).split('\n').map(line => line.trim()).filter(Boolean);
  const subject = message.split('\n', 1)[0];
  const squash = subject.match(SQUASH_SUFFIX);
  const merge = subject.match(MERGE_SUBJECT);
  const number = Number((squash ?? merge)?.[1]);
  const commit = { sha, message, files };
  if (number) {
    commit.pullRequest = {
      sha,
      number,
      title: subject.replace(SQUASH_SUFFIX, ''),
      body: overrideBodies.get(number) ?? message,
      labels: [],
      files,
      baseBranchName: targetBranch,
      headBranchName: merge ? merge[2].trim() : '',
    };
  }
  return commit;
}

/**
 * Yield commits reachable from `ref`, newest first, in the shape
 * release-please's GraphQL iterator returns.
 *
 * - `--date-order` never lists a commit before its children, so the walk can't
 *   reach a release commit before commits that landed after it.
 * - `--diff-merges=first-parent` lists a merge commit's files against its first
 *   parent, as GitHub does. Without it a merge lists no files, and
 *   release-please assigns a file-less commit to every package.
 * - `--no-renames` keeps rename detection from fetching blobs in a blobless
 *   clone. A rename is listed as its old and new paths.
 * - The PR comes from the subject's `(#N)` squash suffix or a
 *   `Merge pull request #N` subject. Its body is the commit message unless the
 *   PR carries a `BEGIN_COMMIT_OVERRIDE` block.
 *
 * Stops `git log` when the caller stops reading.
 */
export async function* gitMergeCommits(ref, { targetBranch, overrideBodies = new Map(), maxResults, cwd } = {}) {
  const args = ['log', ref, '--date-order', '--no-renames', '--diff-merges=first-parent', '--name-only',
    `--format=${RECORD}%H%n%B${MESSAGE_END}`];
  if (maxResults && maxResults < Number.MAX_SAFE_INTEGER) args.push('-n', String(maxResults));
  const child = spawn('git', args, { cwd, timeout: 120_000, stdio: ['ignore', 'pipe', 'inherit'] });
  child.stdout.setEncoding('utf8');
  const exited = new Promise((resolve, reject) => {
    child.on('error', reject);
    child.on('close', code => resolve(code));
  });

  let buffer = '';
  let finished = false;
  try {
    for await (const chunk of child.stdout) {
      buffer += chunk;
      const records = buffer.split(RECORD);
      buffer = records.pop();
      for (const record of records) {
        if (record) yield toCommit(record, targetBranch, overrideBodies);
      }
    }
    if (buffer) yield toCommit(buffer, targetBranch, overrideBodies);
    finished = true;
  } finally {
    if (!finished) {
      child.kill();
      exited.catch(() => {});
    }
  }
  const code = await exited;
  if (code !== 0) throw new Error(`git log exited with code ${code}`);
}
