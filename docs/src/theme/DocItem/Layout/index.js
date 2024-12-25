import React from 'react';
import clsx from 'clsx';
import {useWindowSize} from '@docusaurus/theme-common';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import DocItemPaginator from '@theme/DocItem/Paginator';
import DocVersionBanner from '@theme/DocVersionBanner';
import DocVersionBadge from '@theme/DocVersionBadge';
import DocItemFooter from '@theme/DocItem/Footer';
import DocItemTOCMobile from '@theme/DocItem/TOC/Mobile';
import DocItemTOCDesktop from '@theme/DocItem/TOC/Desktop';
import DocItemContent from '@theme/DocItem/Content';
import DocBreadcrumbs from '@theme/DocBreadcrumbs';
import ContentVisibility from '@theme/ContentVisibility';
import styles from './styles.module.css';
/**
 * Decide if the toc should be rendered, on mobile or desktop viewports
 */
function useDocTOC() {
  const {frontMatter, toc} = useDoc();
  const windowSize = useWindowSize();
  const hidden = frontMatter.hide_table_of_contents;
  const canRender = !hidden && toc.length > 0;
  const mobile = canRender ? <DocItemTOCMobile /> : undefined;
  const desktop =
    canRender && (windowSize === 'desktop' || windowSize === 'ssr') ? (
      <DocItemTOCDesktop />
    ) : undefined;
  return {
    hidden,
    mobile,
    desktop,
  };
}
export default function DocItemLayout({children}) {
  const docTOC = useDocTOC();
  const {metadata, frontMatter} = useDoc();

  "https://github.com/langchain-ai/langchain/blob/master/docs/docs/introduction.ipynb"
  "https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/introduction.ipynb"

  const linkColab = frontMatter.link_colab || (
    metadata.editUrl?.endsWith(".ipynb") 
      ? metadata.editUrl?.replace("https://github.com/langchain-ai/langchain/edit/", "https://colab.research.google.com/github/langchain-ai/langchain/blob/") 
      : null
  );
  const linkGithub = frontMatter.link_github || metadata.editUrl?.replace("/edit/", "/blob/");

  return (
    <div className="row">
      <div className={clsx('col', !docTOC.hidden && styles.docItemCol)}>
        <ContentVisibility metadata={metadata} />
        <DocVersionBanner />
        <div className={styles.docItemContainer}>
          <article>
            <DocBreadcrumbs />
            <DocVersionBadge />
            {docTOC.mobile}
            <div style={{ 
              display: "flex", 
              flexDirection: "column",
              alignItems: "flex-end",
              float: "right",
              marginLeft: 12,
            }}>
              {linkColab && (<a target="_blank" href={linkColab}>
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
              </a>)}
              {linkGithub && (<a href={linkGithub} target="_blank">
                <img src="https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github&logoColor=white" 
                    alt="Open on GitHub" />
              </a>)}
            </div>
            <DocItemContent>{children}</DocItemContent>
            <DocItemFooter />
          </article>
          <DocItemPaginator />
        </div>
      </div>
      {docTOC.desktop && <div className="col col--3">{docTOC.desktop}</div>}
    </div>
  );
}
