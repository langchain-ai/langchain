import React from 'react';
import Paginator from '@theme-original/DocItem/Paginator';
import Feedback from "@theme/Feedback";
import Giscus from "@giscus/react";

export default function PaginatorWrapper(props) {
  return (
    <>
      <Feedback />
      <Giscus
        repo="langchain-ai/langchain"
        repoId="R_kgDOIPDwlg"
        category="Docs Discussions"
        categoryId="DIC_kwDOIPDwls4CjJYb"
        mapping="pathname"
        strict="1"
        reactionsEnabled="0"
        emitMetadata="0"
        inputPosition="bottom"
        theme="preferred_color_scheme"
        lang="en"
        loading="lazy" />
      <Paginator {...props} />
    </>
  );
}
