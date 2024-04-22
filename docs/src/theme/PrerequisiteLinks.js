import React from "react";
import { marked } from "marked";
import DOMPurify from "isomorphic-dompurify";
import Admonition from '@theme/Admonition';

export default function PrerequisiteLinks({ content }) {
  return (
    <Admonition type="info" title="Prerequisites">
      <div style={{ marginTop: "8px" }}>
        This guide will assume familiarity with the following concepts:
      </div>
      <div style={{ marginTop: "16px" }}
        dangerouslySetInnerHTML={{
          __html: DOMPurify.sanitize(marked.parse(content))
        }} 
      />
    </Admonition>
  );
}
