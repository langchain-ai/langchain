/*
 * searchtools.js
 * ~~~~~~~~~~~~~~~~
 *
 * Sphinx JavaScript utilities for the full-text search.
 *
 * :copyright: Copyright 2007-2023 by the Sphinx team, see AUTHORS.
 * :license: BSD, see LICENSE for details.
 *
 */
"use strict";

/**
 * Simple result scoring code.
 */
if (typeof Scorer === "undefined") {
  var Scorer = {
    // Implement the following function to further tweak the score for each result
    // The function takes a result array [docname, title, anchor, descr, score, filename]
    // and returns the new score.
    /*
    score: result => {
      const [docname, title, anchor, descr, score, filename] = result
      return score
    },
    */

    // query matches the full name of an object
    objNameMatch: 11,
    // or matches in the last dotted part of the object name
    objPartialMatch: 6,
    // Additive scores depending on the priority of the object
    objPrio: {
      0: 15, // used to be importantResults
      1: 5, // used to be objectResults
      2: -5, // used to be unimportantResults
    },
    //  Used when the priority is not in the mapping.
    objPrioDefault: 0,

    // query found in title
    title: 15,
    partialTitle: 7,
    // query found in terms
    term: 5,
    partialTerm: 2,
  };
}

const _removeChildren = (element) => {
  while (element && element.lastChild) element.removeChild(element.lastChild);
};

/**
 * See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions#escaping
 */
const _escapeRegExp = (string) =>
  string.replace(/[.*+\-?^${}()|[\]\\]/g, "\\$&"); // $& means the whole matched string

const _displayItem = (item, searchTerms) => {
  const docBuilder = DOCUMENTATION_OPTIONS.BUILDER;
  const docUrlRoot = DOCUMENTATION_OPTIONS.URL_ROOT;
  const docFileSuffix = DOCUMENTATION_OPTIONS.FILE_SUFFIX;
  const docLinkSuffix = DOCUMENTATION_OPTIONS.LINK_SUFFIX;
  const showSearchSummary = DOCUMENTATION_OPTIONS.SHOW_SEARCH_SUMMARY;

  const [docName, title, anchor, descr, score, _filename] = item;

  let listItem = document.createElement("li");
  let requestUrl;
  let linkUrl;
  if (docBuilder === "dirhtml") {
    // dirhtml builder
    let dirname = docName + "/";
    if (dirname.match(/\/index\/$/))
      dirname = dirname.substring(0, dirname.length - 6);
    else if (dirname === "index/") dirname = "";
    requestUrl = docUrlRoot + dirname;
    linkUrl = requestUrl;
  } else {
    // normal html builders
    requestUrl = docUrlRoot + docName + docFileSuffix;
    linkUrl = docName + docLinkSuffix;
  }
  let linkEl = listItem.appendChild(document.createElement("a"));
  linkEl.href = linkUrl + anchor;
  linkEl.dataset.score = score;
  linkEl.innerHTML = title;
  if (descr)
    listItem.appendChild(document.createElement("span")).innerHTML =
      " (" + descr + ")";
  else if (showSearchSummary)
    fetch(requestUrl)
      .then((responseData) => responseData.text())
      .then((data) => {
        if (data)
          listItem.appendChild(
            Search.makeSearchSummary(data, searchTerms)
          );
      });
  Search.output.appendChild(listItem);
};
const _finishSearch = (resultCount) => {
  Search.stopPulse();
  Search.title.innerText = _("Search Results");
  if (!resultCount)
    Search.status.innerText = Documentation.gettext(
      "Your search did not match any documents. Please make sure that all words are spelled correctly and that you've selected enough categories."
    );
  else
    Search.status.innerText = _(
      `Search finished, found ${resultCount} page(s) matching the search query.`
    );
};
const _displayNextItem = (
  results,
  resultCount,
  searchTerms
) => {
  // results left, load the summary and display it
  // this is intended to be dynamic (don't sub resultsCount)
  if (results.length) {
    _displayItem(results.pop(), searchTerms);
    setTimeout(
      () => _displayNextItem(results, resultCount, searchTerms),
      5
    );
  }
  // search finished, update title and status message
  else _finishSearch(resultCount);
};

/**
 * Default splitQuery function. Can be overridden in ``sphinx.search`` with a
 * custom function per language.
 *
 * The regular expression works by splitting the string on consecutive characters
 * that are not Unicode letters, numbers, underscores, or emoji characters.
 * This is the same as ``\W+`` in Python, preserving the surrogate pair area.
 */
if (typeof splitQuery === "undefined") {
  var splitQuery = (query) => query
      .split(/[^\p{Letter}\p{Number}_\p{Emoji_Presentation}]+/gu)
      .filter(term => term)  // remove remaining empty strings
}

/**
 * Search Module
 */
const Search = {
  _index: null,
  _queued_query: null,
  _pulse_status: -1,

  htmlToText: (htmlString) => {
    const htmlElement = new DOMParser().parseFromString(htmlString, 'text/html');
    htmlElement.querySelectorAll(".headerlink").forEach((el) => { el.remove() });
    const docContent = htmlElement.querySelector('[role="main"]');
    if (docContent !== undefined) return docContent.textContent;
    console.warn(
      "Content block not found. Sphinx search tries to obtain it via '[role=main]'. Could you check your theme or template."
    );
    return "";
  },

  init: () => {
    const query = new URLSearchParams(window.location.search).get("q");
    document
      .querySelectorAll('input[name="q"]')
      .forEach((el) => (el.value = query));
    if (query) Search.performSearch(query);
  },

  loadIndex: (url) =>
    (document.body.appendChild(document.createElement("script")).src = url),

  setIndex: (index) => {
    Search._index = index;
    if (Search._queued_query !== null) {
      const query = Search._queued_query;
      Search._queued_query = null;
      Search.query(query);
    }
  },

  hasIndex: () => Search._index !== null,

  deferQuery: (query) => (Search._queued_query = query),

  stopPulse: () => (Search._pulse_status = -1),

  startPulse: () => {
    if (Search._pulse_status >= 0) return;

    const pulse = () => {
      Search._pulse_status = (Search._pulse_status + 1) % 4;
      Search.dots.innerText = ".".repeat(Search._pulse_status);
      if (Search._pulse_status >= 0) window.setTimeout(pulse, 500);
    };
    pulse();
  },

  /**
   * perform a search for something (or wait until index is loaded)
   */
  performSearch: (query) => {
    // create the required interface elements
    const searchText = document.createElement("h2");
    searchText.textContent = _("Searching");
    const searchSummary = document.createElement("p");
    searchSummary.classList.add("search-summary");
    searchSummary.innerText = "";
    const searchList = document.createElement("ul");
    searchList.classList.add("search");

    const out = document.getElementById("search-results");
    Search.title = out.appendChild(searchText);
    Search.dots = Search.title.appendChild(document.createElement("span"));
    Search.status = out.appendChild(searchSummary);
    Search.output = out.appendChild(searchList);

    const searchProgress = document.getElementById("search-progress");
    // Some themes don't use the search progress node
    if (searchProgress) {
      searchProgress.innerText = _("Preparing search...");
    }
    Search.startPulse();

    // index already loaded, the browser was quick!
    if (Search.hasIndex()) Search.query(query);
    else Search.deferQuery(query);
  },

  /**
   * execute search (requires search index to be loaded)
   */
  query: (query) => {
    const filenames = Search._index.filenames;
    const docNames = Search._index.docnames;
    const titles = Search._index.titles;
    const allTitles = Search._index.alltitles;
    const indexEntries = Search._index.indexentries;

    // stem the search terms and add them to the correct list
    const stemmer = new Stemmer();
    const searchTerms = new Set();
    const excludedTerms = new Set();
    const highlightTerms = new Set();
    const objectTerms = new Set(splitQuery(query.toLowerCase().trim()));
    splitQuery(query.trim()).forEach((queryTerm) => {
      const queryTermLower = queryTerm.toLowerCase();

      // maybe skip this "word"
      // stopwords array is from language_data.js
      if (
        stopwords.indexOf(queryTermLower) !== -1 ||
        queryTerm.match(/^\d+$/)
      )
        return;

      // stem the word
      let word = stemmer.stemWord(queryTermLower);
      // select the correct list
      if (word[0] === "-") excludedTerms.add(word.substr(1));
      else {
        searchTerms.add(word);
        highlightTerms.add(queryTermLower);
      }
    });

    if (SPHINX_HIGHLIGHT_ENABLED) {  // set in sphinx_highlight.js
      localStorage.setItem("sphinx_highlight_terms", [...highlightTerms].join(" "))
    }

    // console.debug("SEARCH: searching for:");
    // console.info("required: ", [...searchTerms]);
    // console.info("excluded: ", [...excludedTerms]);

    // array of [docname, title, anchor, descr, score, filename]
    let results = [];
    _removeChildren(document.getElementById("search-progress"));

    const queryLower = query.toLowerCase();
    for (const [title, foundTitles] of Object.entries(allTitles)) {
      if (title.toLowerCase().includes(queryLower) && (queryLower.length >= title.length/2)) {
        for (const [file, id] of foundTitles) {
          let score = Math.round(100 * queryLower.length / title.length)
          results.push([
            docNames[file],
            titles[file] !== title ? `${titles[file]} > ${title}` : title,
            id !== null ? "#" + id : "",
            null,
            score,
            filenames[file],
          ]);
        }
      }
    }

    // search for explicit entries in index directives
    for (const [entry, foundEntries] of Object.entries(indexEntries)) {
      if (entry.includes(queryLower) && (queryLower.length >= entry.length/2)) {
        for (const [file, id] of foundEntries) {
          let score = Math.round(100 * queryLower.length / entry.length)
          results.push([
            docNames[file],
            titles[file],
            id ? "#" + id : "",
            null,
            score,
            filenames[file],
          ]);
        }
      }
    }

    // lookup as object
    objectTerms.forEach((term) =>
      results.push(...Search.performObjectSearch(term, objectTerms))
    );

    // lookup as search terms in fulltext
    results.push(...Search.performTermsSearch(searchTerms, excludedTerms));

    // let the scorer override scores with a custom scoring function
    if (Scorer.score) results.forEach((item) => (item[4] = Scorer.score(item)));

    // now sort the results by score (in opposite order of appearance, since the
    // display function below uses pop() to retrieve items) and then
    // alphabetically
    results.sort((a, b) => {
      const leftScore = a[4];
      const rightScore = b[4];
      if (leftScore === rightScore) {
        // same score: sort alphabetically
        const leftTitle = a[1].toLowerCase();
        const rightTitle = b[1].toLowerCase();
        if (leftTitle === rightTitle) return 0;
        return leftTitle > rightTitle ? -1 : 1; // inverted is intentional
      }
      return leftScore > rightScore ? 1 : -1;
    });

    // remove duplicate search results
    // note the reversing of results, so that in the case of duplicates, the highest-scoring entry is kept
    let seen = new Set();
    results = results.reverse().reduce((acc, result) => {
      let resultStr = result.slice(0, 4).concat([result[5]]).map(v => String(v)).join(',');
      if (!seen.has(resultStr)) {
        acc.push(result);
        seen.add(resultStr);
      }
      return acc;
    }, []);

    results = results.reverse();

    // for debugging
    //Search.lastresults = results.slice();  // a copy
    // console.info("search results:", Search.lastresults);

    // print the results
    _displayNextItem(results, results.length, searchTerms);
  },

  /**
   * search for object names
   */
  performObjectSearch: (object, objectTerms) => {
    const filenames = Search._index.filenames;
    const docNames = Search._index.docnames;
    const objects = Search._index.objects;
    const objNames = Search._index.objnames;
    const titles = Search._index.titles;

    const results = [];

    const objectSearchCallback = (prefix, match) => {
      const name = match[4]
      const fullname = (prefix ? prefix + "." : "") + name;
      const fullnameLower = fullname.toLowerCase();
      if (fullnameLower.indexOf(object) < 0) return;

      let score = 0;
      const parts = fullnameLower.split(".");

      // check for different match types: exact matches of full name or
      // "last name" (i.e. last dotted part)
      if (fullnameLower === object || parts.slice(-1)[0] === object)
        score += Scorer.objNameMatch;
      else if (parts.slice(-1)[0].indexOf(object) > -1)
        score += Scorer.objPartialMatch; // matches in last name

      const objName = objNames[match[1]][2];
      const title = titles[match[0]];

      // If more than one term searched for, we require other words to be
      // found in the name/title/description
      const otherTerms = new Set(objectTerms);
      otherTerms.delete(object);
      if (otherTerms.size > 0) {
        const haystack = `${prefix} ${name} ${objName} ${title}`.toLowerCase();
        if (
          [...otherTerms].some((otherTerm) => haystack.indexOf(otherTerm) < 0)
        )
          return;
      }

      let anchor = match[3];
      if (anchor === "") anchor = fullname;
      else if (anchor === "-") anchor = objNames[match[1]][1] + "-" + fullname;

      const descr = objName + _(", in ") + title;

      // add custom score for some objects according to scorer
      if (Scorer.objPrio.hasOwnProperty(match[2]))
        score += Scorer.objPrio[match[2]];
      else score += Scorer.objPrioDefault;

      results.push([
        docNames[match[0]],
        fullname,
        "#" + anchor,
        descr,
        score,
        filenames[match[0]],
      ]);
    };
    Object.keys(objects).forEach((prefix) =>
      objects[prefix].forEach((array) =>
        objectSearchCallback(prefix, array)
      )
    );
    return results;
  },

  /**
   * search for full-text terms in the index
   */
  performTermsSearch: (searchTerms, excludedTerms) => {
    // prepare search
    const terms = Search._index.terms;
    const titleTerms = Search._index.titleterms;
    const filenames = Search._index.filenames;
    const docNames = Search._index.docnames;
    const titles = Search._index.titles;

    const scoreMap = new Map();
    const fileMap = new Map();

    // perform the search on the required terms
    searchTerms.forEach((word) => {
      const files = [];
      const arr = [
        { files: terms[word], score: Scorer.term },
        { files: titleTerms[word], score: Scorer.title },
      ];
      // add support for partial matches
      if (word.length > 2) {
        const escapedWord = _escapeRegExp(word);
        Object.keys(terms).forEach((term) => {
          if (term.match(escapedWord) && !terms[word])
            arr.push({ files: terms[term], score: Scorer.partialTerm });
        });
        Object.keys(titleTerms).forEach((term) => {
          if (term.match(escapedWord) && !titleTerms[word])
            arr.push({ files: titleTerms[word], score: Scorer.partialTitle });
        });
      }

      // no match but word was a required one
      if (arr.every((record) => record.files === undefined)) return;

      // found search word in contents
      arr.forEach((record) => {
        if (record.files === undefined) return;

        let recordFiles = record.files;
        if (recordFiles.length === undefined) recordFiles = [recordFiles];
        files.push(...recordFiles);

        // set score for the word in each file
        recordFiles.forEach((file) => {
          if (!scoreMap.has(file)) scoreMap.set(file, {});
          scoreMap.get(file)[word] = record.score;
        });
      });

      // create the mapping
      files.forEach((file) => {
        if (fileMap.has(file) && fileMap.get(file).indexOf(word) === -1)
          fileMap.get(file).push(word);
        else fileMap.set(file, [word]);
      });
    });

    // now check if the files don't contain excluded terms
    const results = [];
    for (const [file, wordList] of fileMap) {
      // check if all requirements are matched

      // as search terms with length < 3 are discarded
      const filteredTermCount = [...searchTerms].filter(
        (term) => term.length > 2
      ).length;
      if (
        wordList.length !== searchTerms.size &&
        wordList.length !== filteredTermCount
      )
        continue;

      // ensure that none of the excluded terms is in the search result
      if (
        [...excludedTerms].some(
          (term) =>
            terms[term] === file ||
            titleTerms[term] === file ||
            (terms[term] || []).includes(file) ||
            (titleTerms[term] || []).includes(file)
        )
      )
        break;

      // select one (max) score for the file.
      const score = Math.max(...wordList.map((w) => scoreMap.get(file)[w]));
      // add result to the result list
      results.push([
        docNames[file],
        titles[file],
        "",
        null,
        score,
        filenames[file],
      ]);
    }
    return results;
  },

  /**
   * helper function to return a node containing the
   * search summary for a given text. keywords is a list
   * of stemmed words.
   */
  makeSearchSummary: (htmlText, keywords) => {
    const text = Search.htmlToText(htmlText);
    if (text === "") return null;

    const textLower = text.toLowerCase();
    const actualStartPosition = [...keywords]
      .map((k) => textLower.indexOf(k.toLowerCase()))
      .filter((i) => i > -1)
      .slice(-1)[0];
    const startWithContext = Math.max(actualStartPosition - 120, 0);

    const top = startWithContext === 0 ? "" : "...";
    const tail = startWithContext + 240 < text.length ? "..." : "";

    let summary = document.createElement("p");
    summary.classList.add("context");
    summary.textContent = top + text.substr(startWithContext, 240).trim() + tail;

    return summary;
  },
};

_ready(Search.init);
