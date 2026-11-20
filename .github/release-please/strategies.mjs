import { createRequire } from 'node:module';
import { registerReleaseType, registerVersioningStrategy } from 'release-please';

const require = createRequire(import.meta.url);
const { Python } = require('release-please/build/src/strategies/python');
const { DefaultVersioningStrategy } = require('release-please/build/src/versioning-strategies/default');

export class LangChainVersioningStrategy extends DefaultVersioningStrategy {
  determineReleaseType(version, commits) {
    const patchFeatures = version.isPreMajor ? commits : commits.map(commit => (
      commit.type === 'feat' || commit.type === 'feature' ? { ...commit, type: 'fix' } : commit
    ));
    return super.determineReleaseType(version, patchFeatures);
  }
}

export class LangChainPython extends Python {
  async extraFileUpdates(version, versionsMap, dateFormat) {
    const updates = await super.extraFileUpdates(version, versionsMap, dateFormat);
    const pythonFiles = new Set(this.extraFiles
      .filter(file => typeof file === 'string' && file.endsWith('.py'))
      .map(file => this.addPath(file)));
    for (const update of updates) {
      if (pythonFiles.has(update.path)) {
        update.updater = {
          updateContent: content => content.replace(
            /^((?:__version__|VERSION)\s*=\s*["'])[^"'\n]+(["'])/m,
            `$1${version}$2`,
          ),
        };
      }
    }
    return updates;
  }
}

registerReleaseType('python', options => new LangChainPython(options));
registerVersioningStrategy('langchain', options => new LangChainVersioningStrategy(options));
