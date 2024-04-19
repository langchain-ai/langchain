/* eslint-disable react/jsx-props-no-spreading, react/destructuring-assignment */
import React from "react";

const BASE_GIT_URL = "https://api.github.com/repos/langchain-ai/langchain/commits?path=docs"
const LAST_UPDATED_ELEMENT_ID = "lc_last_updated"
const INVALID_DATE_STRING = "Invalid date"
/**
 * 
 * @param {Array<string>} urls 
 * @returns {Promise<string | null>} The formatted date string or null if not found
 */
const fetchUrls = async (urls) => {
  try {
    const allResponses = await Promise.allSettled(urls.map(url => fetch(url)))
    const allOkResponses = allResponses.filter(({ ok }) => ok);
    const allData = await Promise.allSettled(allOkResponses.map(({ value }) => value.json()));
    /** @type {null | string} */
    let formattedDate = null;
    allData.forEach((item) => {
      if (formattedDate && formattedDate !== INVALID_DATE_STRING || !item || item.length === 0 || !item[0]?.commit?.author?.date) return;
      const lastCommitDate = new Date(item[0]?.commit?.author?.date);
      formattedDate = lastCommitDate.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
    })
    if (formattedDate !== INVALID_DATE_STRING) {
      return formattedDate;
    }
  } catch (_) {
    // no-op
  }
  return null;
}

const getAllPossibleUrls = (currentPath) => {
  currentPath = currentPath.endsWith("/") ? currentPath.slice(0, -1) : currentPath;
  return {
    notebookPath: `${BASE_GIT_URL}${currentPath}.ipynb`,
    notebookIndexPath: `${BASE_GIT_URL}${currentPath}/index.ipynb`,
    mdPath: `${BASE_GIT_URL}${currentPath}.md`,
    mdIndexPath: `${BASE_GIT_URL}${currentPath}/index.md`,
    mdxPath: `${BASE_GIT_URL}${currentPath}.mdx`,
    mdxIndexPath: `${BASE_GIT_URL}${currentPath}/index.mdx`,
  }
}

/**
 * NOTE: This component file should NEVER be updated as it overrides
 * the default docusaurus LastUpdated component.
 */
export default function LastUpdated() {
  const [lastUpdatedDate, setLastUpdatedDate] = React.useState(null);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      let currentPath = window.location.pathname;
      const allUrls = getAllPossibleUrls(currentPath);
      const { notebookPath, notebookIndexPath, ...rest } = allUrls;

      fetchUrls([notebookPath, notebookIndexPath])
        .then((date) => {
          if (date) {
            setLastUpdatedDate(date);
          } else {
            fetchUrls(rest)
              .then((date) => {
                if (date) {
                  setLastUpdatedDate(date);
                }
              });
          }
        });
    } catch (_) {
      // no-op
    }
  }, [])

  if (!lastUpdatedDate) return null;

  return (
    <div style={{ display: 'flex', justifyContent: 'flex-end' }} id={LAST_UPDATED_ELEMENT_ID}>
      <p style={{ fontSize: 13.3333 }}><i>Last updated on <strong>{lastUpdatedDate}</strong></i></p>
    </div>
  );
}
