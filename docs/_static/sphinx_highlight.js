/* Highlighting utilities for Sphinx HTML documentation. */
"use strict";

const SPHINX_HIGHLIGHT_ENABLED = true

/**
 * highlight a given string on a node by wrapping it in
 * span elements with the given class name.
 */
const _highlight = (node, addItems, text, className) => {
  if (node.nodeType === Node.TEXT_NODE) {
    const val = node.nodeValue;
    const parent = node.parentNode;
    const pos = val.toLowerCase().indexOf(text);
    if (
      pos >= 0 &&
      !parent.classList.contains(className) &&
      !parent.classList.contains("nohighlight")
    ) {
      let span;

      const closestNode = parent.closest("body, svg, foreignObject");
      const isInSVG = closestNode && closestNode.matches("svg");
      if (isInSVG) {
        span = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
      } else {
        span = document.createElement("span");
        span.classList.add(className);
      }

      span.appendChild(document.createTextNode(val.substr(pos, text.length)));
      parent.insertBefore(
        span,
        parent.insertBefore(
          document.createTextNode(val.substr(pos + text.length)),
          node.nextSibling
        )
      );
      node.nodeValue = val.substr(0, pos);

      if (isInSVG) {
        const rect = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "rect"
        );
        const bbox = parent.getBBox();
        rect.x.baseVal.value = bbox.x;
        rect.y.baseVal.value = bbox.y;
        rect.width.baseVal.value = bbox.width;
        rect.height.baseVal.value = bbox.height;
        rect.setAttribute("class", className);
        addItems.push({ parent: parent, target: rect });
      }
    }
  } else if (node.matches && !node.matches("button, select, textarea")) {
    node.childNodes.forEach((el) => _highlight(el, addItems, text, className));
  }
};
const _highlightText = (thisNode, text, className) => {
  let addItems = [];
  _highlight(thisNode, addItems, text, className);
  addItems.forEach((obj) =>
    obj.parent.insertAdjacentElement("beforebegin", obj.target)
  );
};

/**
 * Small JavaScript module for the documentation.
 */
const SphinxHighlight = {

  /**
   * highlight the search words provided in localstorage in the text
   */
  highlightSearchWords: () => {
    if (!SPHINX_HIGHLIGHT_ENABLED) return;  // bail if no highlight

    // get and clear terms from localstorage
    const url = new URL(window.location);
    const highlight =
        localStorage.getItem("sphinx_highlight_terms")
        || url.searchParams.get("highlight")
        || "";
    localStorage.removeItem("sphinx_highlight_terms")
    url.searchParams.delete("highlight");
    window.history.replaceState({}, "", url);

    // get individual terms from highlight string
    const terms = highlight.toLowerCase().split(/\s+/).filter(x => x);
    if (terms.length === 0) return; // nothing to do

    // There should never be more than one element matching "div.body"
    const divBody = document.querySelectorAll("div.body");
    const body = divBody.length ? divBody[0] : document.querySelector("body");
    window.setTimeout(() => {
      terms.forEach((term) => _highlightText(body, term, "highlighted"));
    }, 10);

    const searchBox = document.getElementById("searchbox");
    if (searchBox === null) return;
    searchBox.appendChild(
      document
        .createRange()
        .createContextualFragment(
          '<p class="highlight-link">' +
            '<a href="javascript:SphinxHighlight.hideSearchWords()">' +
            _("Hide Search Matches") +
            "</a></p>"
        )
    );
  },

  /**
   * helper function to hide the search marks again
   */
  hideSearchWords: () => {
    document
      .querySelectorAll("#searchbox .highlight-link")
      .forEach((el) => el.remove());
    document
      .querySelectorAll("span.highlighted")
      .forEach((el) => el.classList.remove("highlighted"));
    localStorage.removeItem("sphinx_highlight_terms")
  },

  initEscapeListener: () => {
    // only install a listener if it is really needed
    if (!DOCUMENTATION_OPTIONS.ENABLE_SEARCH_SHORTCUTS) return;

    document.addEventListener("keydown", (event) => {
      // bail for input elements
      if (BLACKLISTED_KEY_CONTROL_ELEMENTS.has(document.activeElement.tagName)) return;
      // bail with special keys
      if (event.shiftKey || event.altKey || event.ctrlKey || event.metaKey) return;
      if (DOCUMENTATION_OPTIONS.ENABLE_SEARCH_SHORTCUTS && (event.key === "Escape")) {
        SphinxHighlight.hideSearchWords();
        event.preventDefault();
      }
    });
  },
};

_ready(SphinxHighlight.highlightSearchWords);
_ready(SphinxHighlight.initEscapeListener);
