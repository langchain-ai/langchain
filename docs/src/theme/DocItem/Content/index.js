import React from 'react';
import Content from '@theme-original/DocItem/Content';

export default function ContentWrapper(props) {
  console.log(props);
  const { siteConfig } = useDocusaurusContext();
  console.log(siteConfig);

  return (
    <>
      <div style={{float: "right", display: "flex", flexDirection: "column", alignItems: "flex-end"}}>
        <a target="_blank" href="https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_monitoring/model_monitoring.ipynb">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
        </a>
        <a target="_blank" href="https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/model_monitoring/model_monitoring.ipynb">
          <img src="https://img.shields.io/badge/Open_in_Github-gray?logo=github" alt="Open in Github"/>
        </a>
      </div>

      <Content {...props} />
    </>
  );
}
