/* eslint-disable react/jsx-props-no-spreading, react/destructuring-assignment */
import React from "react";

const BASE_GIT_URL = "https://api.github.com/repos/langchain-ai/langchain/commits?path=docs"

const LAST_UPDATED_ELEMENT_ID = "lc_last_updated"

const fetchUrl = async (url) => {
  try {
    const res = await fetch(url)
    if (!res.ok) return null;
    const json = await res.json();
    if (!json || json.length === 0 || !json[0]?.commit?.author?.date) return null;
    const lastCommitDate = new Date(json[0]?.commit?.author?.date);
    const formattedDate = lastCommitDate.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
    if (formattedDate !== "Invalid Date") {
      return formattedDate;
    }
  } catch (_) {
    // no-op
  }
  return null;
}

/**
 * NOTE: This component file can NOT be named `LastUpdated` as it
 * conflicts with the built-in Docusaurus component, and will override it.
 */
export default function LastUpdatedComponent() {
  const [lastUpdatedDate, setLastUpdatedDate] = React.useState(null);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      let currentPath = window.location.pathname;
      if (currentPath.endsWith("/")) {
        // strip the trailing slash
        currentPath = currentPath.slice(0, -1);
      }
      const apiUrl = `${BASE_GIT_URL}${currentPath}.ipynb`
      const apiUrlWithIndex = `${BASE_GIT_URL}${currentPath}/index.ipynb`

      fetchUrl(apiUrl)
        .then((date) => {
          if (date) {
            setLastUpdatedDate(date);
          } else {
            fetchUrl(apiUrlWithIndex)
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
