import { appendFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

import { GitHub, Manifest, VERSION } from 'release-please';

import './strategies.mjs';
import { fetchOverrideBodies, gitMergeCommits } from './history.mjs';

const require = createRequire(import.meta.url);
const { CommitExclude } = require('release-please/build/src/util/commit-exclude');
const { TagName } = require('release-please/build/src/util/tag-name');

const CONFIG_FILE = 'release-please-config.json';
const MANIFEST_FILE = '.release-please-manifest.json';

export function assertPatched() {
  const commit = { sha: 'x', message: 'chore: relock', files: ['pkg/uv.lock'] };
  const kept = new CommitExclude({ pkg: { excludePaths: ['pkg/uv.lock'] } })
    .excludeCommits({ pkg: [commit] });
  if (kept.pkg.length !== 0) {
    throw new Error('the commit-exclude.js change in the release-please patch is not applied: exclude-paths ignores file paths');
  }

  const tag = TagName.parse('langchain-core==1.8.4');
  if (tag?.component !== 'langchain-core' || tag.separator !== '==') {
    throw new Error('the tag-name.js change in the release-please patch is not applied: `==` tags parse with the wrong component');
  }
}

function setOutput(name, value) {
  const file = process.env.GITHUB_OUTPUT;
  if (!file) return;
  const text = typeof value === 'string' ? value : JSON.stringify(value);
  const delimiter = `ghadelimiter_${randomUUID()}`;
  appendFileSync(file, `${name}<<${delimiter}\n${text}\n${delimiter}\n`);
}

export function writePullRequestOutputs(candidates) {
  const prs = candidates.filter(pr => pr !== undefined);
  setOutput('prs_created', String(prs.length > 0));
  setOutput('prs', prs);
  if (prs.length) setOutput('pr', prs[0]);
}

export async function main() {
  assertPatched();
  console.log(`Running release-please version: ${VERSION}`);

  const { GITHUB_TOKEN: token, GITHUB_REPOSITORY: repository } = process.env;
  if (!token) throw new Error('GITHUB_TOKEN is not set');
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository || '')) {
    throw new Error('GITHUB_REPOSITORY must be owner/repo');
  }
  const [owner, repo] = repository.split('/');
  const github = await GitHub.create({ owner, repo, token });
  const targetBranch = github.repository.defaultBranch;

  // Walk history from a local clone instead of release-please's GraphQL
  // iterator: 10 commits per request makes the walk most of the run, it fails
  // on GitHub timeouts, and its 500-commit cap hides commits of packages that
  // haven't released in longer than that. Fetch first so the walk starts at the
  // same branch tip the API reads.
  const ref = `refs/remotes/origin/${targetBranch}`;
  execFileSync('git', ['fetch', '--quiet', '--filter=blob:none', 'origin', `+refs/heads/${targetBranch}:${ref}`],
    { stdio: 'inherit', timeout: 120_000 });
  const overrideBodies = await fetchOverrideBodies({ owner, repo, token });
  console.log(`Found ${overrideBodies.size} merged PRs with BEGIN_COMMIT_OVERRIDE`);
  github.mergeCommitIterator = (branch, options = {}) => {
    if (branch !== targetBranch) throw new Error(`Unexpected commit walk on ${branch}`);
    return gitMergeCommits(ref, { targetBranch: branch, overrideBodies, maxResults: options.maxResults });
  };

  const manifest = await Manifest.fromManifest(github, targetBranch, CONFIG_FILE, MANIFEST_FILE);
  // The default cap (500) is a cost guard for the GraphQL walk. Locally the
  // walk is cheap, and release-please stops once it reaches every package's
  // last release commit.
  manifest.commitSearchDepth = Number.MAX_SAFE_INTEGER;

  if (process.env.DRY_RUN === 'true') {
    const candidates = await manifest.buildPullRequests();
    console.log(JSON.stringify(candidates.map(pr => ({
      title: pr.title.toString(),
      branch: pr.headRefName,
      body: pr.body.toString(),
    })), null, 2));
    writePullRequestOutputs([]);
    return;
  }

  writePullRequestOutputs(await manifest.createPullRequests());
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(err => {
    const message = String(err.message).replaceAll(process.env.GITHUB_TOKEN || '\0', '[REDACTED]');
    console.log(`::error::release-please failed: ${message}`);
    process.exitCode = 1;
  });
}
