/*
 * doctools.js
 * ~~~~~~~~~~~
 *
 * Base JavaScript utilities for all Sphinx HTML documentation.
 *
 * :copyright: Copyright 2007-2023 by the Sphinx team, see AUTHORS.
 * :license: BSD, see LICENSE for details.
 *
 */
"use strict";

const BLACKLISTED_KEY_CONTROL_ELEMENTS = new Set([
  "TEXTAREA",
  "INPUT",
  "SELECT",
  "BUTTON",
]);

const _ready = (callback) => {
  if (document.readyState !== "loading") {
    callback();
  } else {
    document.addEventListener("DOMContentLoaded", callback);
  }
};

/**
 * Small JavaScript module for the documentation.
 */
const Documentation = {
  init: () => {
    Documentation.initDomainIndexTable();
    Documentation.initOnKeyListeners();
  },

  /**
   * i18n support
   */
  TRANSLATIONS: {},
  PLURAL_EXPR: (n) => (n === 1 ? 0 : 1),
  LOCALE: "unknown",

  // gettext and ngettext don't access this so that the functions
  // can safely bound to a different name (_ = Documentation.gettext)
  gettext: (string) => {
    const translated = Documentation.TRANSLATIONS[string];
    switch (typeof translated) {
      case "undefined":
        return string; // no translation
      case "string":
        return translated; // translation exists
      default:
        return translated[0]; // (singular, plural) translation tuple exists
    }
  },

  ngettext: (singular, plural, n) => {
    const translated = Documentation.TRANSLATIONS[singular];
    if (typeof translated !== "undefined")
      return translated[Documentation.PLURAL_EXPR(n)];
    return n === 1 ? singular : plural;
  },

  addTranslations: (catalog) => {
    Object.assign(Documentation.TRANSLATIONS, catalog.messages);
    Documentation.PLURAL_EXPR = new Function(
      "n",
      `return (${catalog.plural_expr})`
    );
    Documentation.LOCALE = catalog.locale;
  },

  /**
   * helper function to focus on search bar
   */
  focusSearchBar: () => {
    document.querySelectorAll("input[name=q]")[0]?.focus();
  },

  /**
   * Initialise the domain index toggle buttons
   */
  initDomainIndexTable: () => {
    const toggler = (el) => {
      const idNumber = el.id.substr(7);
      const toggledRows = document.querySelectorAll(`tr.cg-${idNumber}`);
      if (el.src.substr(-9) === "minus.png") {
        el.src = `${el.src.substr(0, el.src.length - 9)}plus.png`;
        toggledRows.forEach((el) => (el.style.display = "none"));
      } else {
        el.src = `${el.src.substr(0, el.src.length - 8)}minus.png`;
        toggledRows.forEach((el) => (el.style.display = ""));
      }
    };

    const togglerElements = document.querySelectorAll("img.toggler");
    togglerElements.forEach((el) =>
      el.addEventListener("click", (event) => toggler(event.currentTarget))
    );
    togglerElements.forEach((el) => (el.style.display = ""));
    if (DOCUMENTATION_OPTIONS.COLLAPSE_INDEX) togglerElements.forEach(toggler);
  },

  initOnKeyListeners: () => {
    // only install a listener if it is really needed
    if (
      !DOCUMENTATION_OPTIONS.NAVIGATION_WITH_KEYS &&
      !DOCUMENTATION_OPTIONS.ENABLE_SEARCH_SHORTCUTS
    )
      return;

    document.addEventListener("keydown", (event) => {
      // bail for input elements
      if (BLACKLISTED_KEY_CONTROL_ELEMENTS.has(document.activeElement.tagName)) return;
      // bail with special keys
      if (event.altKey || event.ctrlKey || event.metaKey) return;

      if (!event.shiftKey) {
        switch (event.key) {
          case "ArrowLeft":
            if (!DOCUMENTATION_OPTIONS.NAVIGATION_WITH_KEYS) break;

            const prevLink = document.querySelector('link[rel="prev"]');
            if (prevLink && prevLink.href) {
              window.location.href = prevLink.href;
              event.preventDefault();
            }
            break;
          case "ArrowRight":
            if (!DOCUMENTATION_OPTIONS.NAVIGATION_WITH_KEYS) break;

            const nextLink = document.querySelector('link[rel="next"]');
            if (nextLink && nextLink.href) {
              window.location.href = nextLink.href;
              event.preventDefault();
            }
            break;
        }
      }

      // some keyboard layouts may need Shift to get /
      switch (event.key) {
        case "/":
          if (!DOCUMENTATION_OPTIONS.ENABLE_SEARCH_SHORTCUTS) break;
          Documentation.focusSearchBar();
          event.preventDefault();
      }
    });
  },
};

// quick alias for translations
const _ = Documentation.gettext;

_ready(Documentation.init);
