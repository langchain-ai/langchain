import React from 'react';
import Paginator from '@theme-original/DocItem/Paginator';
import Feedback from "@theme/Feedback";

export default function PaginatorWrapper(props) {
  return (
    <>
      <Feedback />
      <script src="https://giscus.app/client.js"
        data-repo="langchain-ai/langchain"
        data-repo-id="R_kgDOIPDwlg"
        data-category="Docs Discussions"
        data-category-id="DIC_kwDOIPDwls4CjJYb"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="preferred_color_scheme"
        data-lang="en"
        data-loading="lazy"
        crossorigin="anonymous"
        async>
      </script>
      <Paginator {...props} />
    </>
  );
}
