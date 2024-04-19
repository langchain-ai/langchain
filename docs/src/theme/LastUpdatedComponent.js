/* eslint-disable react/jsx-props-no-spreading, react/destructuring-assignment */
import React from "react";

const BASE_GIT_URL = "https://api.github.com/repos/langchain-ai/langchain/commits?path=docs"

const LAST_UPDATED_ELEMENT_ID = "lc_last_updated"

/**
 * NOTE: This component file can NOT be named `LastUpdated` as it
 * conflicts with the built-in Docusaurus component, and will override it.
 */
export default function LastUpdatedComponent() {
  const [lastUpdatedDate, setLastUpdatedDate] = React.useState(null);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const currentPath = window.location.pathname;
      let apiUrl = ""
      if (currentPath.endsWith("/")) {
        apiUrl = `${BASE_GIT_URL}${currentPath}index.ipynb`;
      } else {
        apiUrl = `${BASE_GIT_URL}${currentPath}.ipynb`;
      }

      fetch(apiUrl)
        .then((response) => response.json())
        .then((data) => {
          if (!data || data.length === 0 || !data[0]?.commit?.author?.date) return;
          const lastCommitDate = new Date(data[0]?.commit?.author?.date);
          const formattedDate = lastCommitDate.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
          });
          if (formattedDate !== "Invalid Date") {
            setLastUpdatedDate(formattedDate);
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
