import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { Manifest, setLogger } from 'release-please';

import { assertPatched, writePullRequestOutputs } from './run.mjs';
import { LangChainVersioningStrategy } from './strategies.mjs';

const require = createRequire(import.meta.url);
const { Version } = require('release-please/build/src/version');
const { CommitExclude } = require('release-please/build/src/util/commit-exclude');
const root = fileURLToPath(new URL('../../', import.meta.url));
const config = JSON.parse(readFileSync(join(root, 'release-please-config.json'), 'utf8'));
const versions = JSON.parse(readFileSync(join(root, '.release-please-manifest.json'), 'utf8'));
const logger = { debug() {}, info() {}, warn() {}, error() {} };
setLogger(logger);

const commit = (type, breaking = false, notes = []) => ({ type, breaking, notes });

function githubFor(commits = []) {
  return {
    repository: { owner: 'langchain-ai', repo: 'langchain', defaultBranch: 'master' },
    getFileJson: async (path, branch) => {
      assert.equal(branch, 'master');
      return JSON.parse(readFileSync(join(root, path), 'utf8'));
    },
    getFileContentsOnBranch: async (path, branch) => {
      assert.equal(branch, 'master');
      return { parsedContent: readFileSync(join(root, path), 'utf8') };
    },
    findFilesByFilenameAndRef: async (filename, branch, path) => {
      assert.equal(branch, 'master');
      return filename === 'version.py' && path === 'libs/core' ? ['langchain_core/version.py'] : [];
    },
    async *releaseIterator() {
      for (const [path, version] of Object.entries(versions)) {
        yield { tagName: `${config.packages[path].component}==${version}`, sha: 'baseline', notes: '' };
      }
    },
    async *mergeCommitIterator(branch) {
      assert.equal(branch, 'master');
      yield* commits;
      yield { sha: 'baseline', message: 'release(core): baseline', files: [] };
      yield { sha: 'old', message: 'feat(core): already released', files: ['libs/core/langchain_core/version.py'] };
    },
  };
}

async function manifestFor(commits) {
  const manifest = await Manifest.fromManifest(githubFor(commits), 'master', undefined, undefined, { logger });
  return manifest;
}

test('vendor patch handles exact files and double-equals tags', () => {
  assertPatched();
});

test('features bump patch after 1.0 without suppressing breaking changes or Release-As', () => {
  const strategy = new LangChainVersioningStrategy();
  const version = Version.parse('1.8.4');
  assert.equal(strategy.bump(version, [commit('feat')]).toString(), '1.8.5');
  assert.equal(strategy.bump(version, [commit('feature')]).toString(), '1.8.5');
  assert.equal(strategy.bump(version, [commit('fix')]).toString(), '1.8.5');
  assert.equal(strategy.bump(version, [commit('feat', true)]).toString(), '2.0.0');
  assert.equal(strategy.bump(version, [commit('feat', false, [{ title: 'RELEASE AS', text: '1.9.0' }])]).toString(), '1.9.0');
});

test('pre-1.0 features bump patch and breaking changes bump minor', () => {
  const strategy = new LangChainVersioningStrategy({ bumpMinorPreMajor: true, bumpPatchForMinorPreMajor: true });
  const version = Version.parse('0.3.0');
  assert.equal(strategy.bump(version, [commit('feat')]).toString(), '0.3.1');
  assert.equal(strategy.bump(version, [commit('feat', true)]).toString(), '0.4.0');
});

test('outputs always include a JSON prs array, including no changes', () => {
  const directory = mkdtempSync(join(tmpdir(), 'release-please-outputs-'));
  const previous = process.env.GITHUB_OUTPUT;
  try {
    for (const [index, candidates] of [[], [undefined], [{ number: 42 }, undefined]].entries()) {
      const file = join(directory, String(index));
      process.env.GITHUB_OUTPUT = file;
      writePullRequestOutputs(candidates);
      const output = readFileSync(file, 'utf8');
      const prs = output.match(/^prs<<([^\n]+)\n([^\n]+)\n\1$/m);
      assert.ok(prs);
      assert.deepEqual(JSON.parse(prs[2]), candidates.filter(pr => pr !== undefined));
      assert.match(output, new RegExp(`prs_created<<[^\\n]+\\n${index === 2}\\n`));
      assert.equal(/^pr<</m.test(output), index === 2);
    }
  } finally {
    if (previous === undefined) delete process.env.GITHUB_OUTPUT;
    else process.env.GITHUB_OUTPUT = previous;
    rmSync(directory, { recursive: true, force: true });
  }
});

test('config covers all publisher dropdown packages and only existing version files', () => {
  const workflow = readFileSync(join(root, '.github/workflows/_release.yml'), 'utf8');
  const options = workflow.match(/        options:\n([\s\S]*?)      working-directory-override:/)[1];
  const names = [...options.matchAll(/^          - ([a-z0-9_-]+)$/gm)].map(match => match[1]);
  const main = new Set(['core', 'langchain', 'langchain_v1', 'text-splitters', 'standard-tests', 'model-profiles']);
  const paths = names.map(name => `libs/${main.has(name) ? '' : 'partners/'}${name}`).sort();
  assert.deepEqual(Object.keys(config.packages).sort(), paths);
  assert.deepEqual(Object.keys(versions).sort(), paths);
  assert.equal(config['draft-pull-request'], true);
  assert.equal(config['separate-pull-requests'], true);
  assert.equal(config['skip-github-release'], true);
  assert.equal(config['tag-separator'], '==');
  assert.equal(config['include-v-in-tag'], false);
  for (const [path, pkg] of Object.entries(config.packages)) {
    const project = readFileSync(join(root, path, 'pyproject.toml'), 'utf8');
    assert.match(project, new RegExp(`^name = "${pkg.component}"$`, 'm'));
    for (const file of pkg['extra-files']) assert.ok(readFileSync(join(root, path, file), 'utf8'));
    const excluded = new CommitExclude({ [path]: { excludePaths: pkg['exclude-paths'] } });
    assert.deepEqual(excluded.excludeCommits({ [path]: [
      { sha: 'test', files: [`${path}/tests/unit_tests/test_module.py`] },
      { sha: 'lock', files: [`${path}/uv.lock`] },
      { sha: 'source', files: [`${path}/tests/test_module.py`, `${path}/package/module.py`] },
    ] })[path].map(commit => commit.sha), ['source']);
  }
});

test('released history, tests, and lockfiles do not generate release PRs', async () => {
  const manifest = await manifestFor([
    { sha: 'test', message: 'fix(core): test helper', files: ['libs/core/tests/test_module.py'] },
    { sha: 'lock', message: 'fix(openai): relock', files: ['libs/partners/openai/uv.lock'] },
  ]);
  assert.deepEqual(await manifest.buildPullRequests(), []);
});

test('each package builds a draft patch PR with correct version updates and no lockfile changes', async () => {
  for (const [path, pkg] of Object.entries(config.packages)) {
    const file = pkg['extra-files'].find(file => file.endsWith('.py')) ?? 'pyproject.toml';
    const manifest = await manifestFor([
      { sha: 'new', message: `feat(${pkg.component}): add capability (#123)`, files: [`${path}/${file}`] },
    ]);
    const [candidate, ...others] = await manifest.buildPullRequests();
    assert.deepEqual(others, []);
    const pr = candidate;
    const [major, minor, patch] = versions[path].split('.').map(Number);
    const next = `${major}.${minor}.${patch + 1}`;
    assert.equal(pr.draft, true);
    assert.equal(pr.title.toString(), `release(${pkg.component}): ${next}`);
    assert.equal(pr.headRefName, `release-please--branches--master--components--${pkg.component}`);
    assert.ok(pr.body.toString().includes('add capability'));
    assert.ok(!pr.updates.some(update => update.path.endsWith('uv.lock')));
    for (const extra of pkg['extra-files']) {
      const update = pr.updates.find(update => update.path === `${path}/${extra}`);
      assert.ok(update, `${path}/${extra} has an updater`);
      const content = readFileSync(join(root, update.path), 'utf8');
      const updated = update.updater.updateContent(content, logger);
      const expression = extra.endsWith('.py')
        ? /^(?:__version__|VERSION)\s*=\s*["']([^"']+)["']/m
        : /^version = "([^"]+)"/m;
      assert.equal(updated.match(expression)?.[1], next, update.path);
    }
    const manifestUpdate = pr.updates.find(update => update.path === '.release-please-manifest.json');
    const changedVersions = JSON.parse(manifestUpdate.updater.updateContent(JSON.stringify(versions)));
    assert.deepEqual(changedVersions, { ...versions, [path]: next });
  }
});


test('tag fallback stops at the release baseline when older releases are outside the release search window', async () => {
  const github = githubFor();
  github.releaseIterator = async function* () {};
  github.tagIterator = async function* () {
    for (const [path, version] of Object.entries(versions)) {
      yield { name: `${config.packages[path].component}==${version}`, sha: 'baseline' };
    }
  };
  const manifest = await Manifest.fromManifest(github, 'master', undefined, undefined, { logger });
  assert.deepEqual(await manifest.buildPullRequests(), []);
});

test('a multi-package change produces separate draft PRs', async () => {
  const manifest = await manifestFor([
    { sha: 'new', message: 'fix(core): shared behavior', files: ['libs/core/langchain_core/version.py', 'libs/partners/openai/langchain_openai/_version.py'] },
  ]);
  const prs = await manifest.buildPullRequests();
  assert.equal(prs.length, 2);
  assert.ok(prs.every(pr => pr.draft));
  assert.equal(new Set(prs.map(pr => pr.headRefName)).size, 2);
});


test('each package excludes its own already-released history while walking to older package releases', async () => {
  const github = githubFor();
  github.releaseIterator = async function* () {
    for (const [path, version] of Object.entries(versions)) {
      yield {
        tagName: `${config.packages[path].component}==${version}`,
        sha: path === 'libs/partners/qdrant' ? 'older-baseline' : 'baseline',
      };
    }
  };
  github.mergeCommitIterator = async function* () {
    yield { sha: 'new', message: 'feat(qdrant): new capability', files: ['libs/partners/qdrant/langchain_qdrant/_version.py'] };
    yield { sha: 'baseline', message: 'release(core): baseline', files: ['libs/core/pyproject.toml'] };
    yield { sha: 'old-core', message: 'feat(core): already released', files: ['libs/core/langchain_core/version.py'] };
    yield { sha: 'older-baseline', message: 'release(qdrant): baseline', files: ['libs/partners/qdrant/pyproject.toml'] };
    yield { sha: 'old-qdrant', message: 'feat(qdrant): already released', files: ['libs/partners/qdrant/langchain_qdrant/_version.py'] };
  };
  const manifest = await Manifest.fromManifest(github, 'master', undefined, undefined, { logger });
  const prs = await manifest.buildPullRequests();
  assert.equal(prs.length, 1);
  assert.match(prs[0].title.toString(), /^release\(langchain-qdrant\): /);
  assert.ok(prs[0].body.toString().includes('new capability'));
  assert.ok(!prs[0].body.toString().includes('already released'));
});
