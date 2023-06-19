/* eslint-disable prefer-template */
/* eslint-disable no-param-reassign */
// eslint-disable-next-line import/no-extraneous-dependencies
const babel = require("@babel/core");
const path = require("path");
const fs = require("fs");

/**
 *
 * @param {string|Buffer} content Content of the resource file
 * @param {object} [map] SourceMap data consumable by https://github.com/mozilla/source-map
 * @param {any} [meta] Meta data, could be anything
 */
async function webpackLoader(content, map, meta) {
  const cb = this.async();

  if (!this.resourcePath.endsWith(".ts")) {
    cb(null, JSON.stringify({ content, imports: [] }), map, meta);
    return;
  }

  try {
    const result = await babel.parseAsync(content, {
      sourceType: "module",
      filename: this.resourcePath,
    });

    const imports = [];

    result.program.body.forEach((node) => {
      if (node.type === "ImportDeclaration") {
        const source = node.source.value;

        if (!source.startsWith("langchain")) {
          return;
        }

        node.specifiers.forEach((specifier) => {
          if (specifier.type === "ImportSpecifier") {
            const local = specifier.local.name;
            const imported = specifier.imported.name;
            imports.push({ local, imported, source });
          } else {
            throw new Error("Unsupported import type");
          }
        });
      }
    });

    imports.forEach((imp) => {
      const { imported, source } = imp;
      const moduleName = source.split("/").slice(1).join("_");
      const docsPath = path.resolve(__dirname, "docs", "api", moduleName);
      const available = fs.readdirSync(docsPath, { withFileTypes: true });
      const found = available.find(
        (dirent) =>
          dirent.isDirectory() &&
          fs.existsSync(path.resolve(docsPath, dirent.name, imported + ".md"))
      );
      if (found) {
        imp.docs =
          "/" + path.join("docs", "api", moduleName, found.name, imported);
      } else {
        throw new Error(
          `Could not find docs for ${source}.${imported} in docs/api/`
        );
      }
    });

    cb(null, JSON.stringify({ content, imports }), map, meta);
  } catch (err) {
    cb(err);
  }
}

module.exports = webpackLoader;
