/* eslint-disable react/jsx-props-no-spreading */
import React from "react";
import CodeBlock from "@theme-original/CodeBlock";

function Imports({ imports }) {
  return (
    <div
      style={{
        paddingTop: "1.3rem",
        background: "var(--prism-background-color)",
        color: "var(--prism-color)",
        marginTop: "calc(-1 * var(--ifm-leading) - 5px)",
        marginBottom: "var(--ifm-leading)",
        boxShadow: "var(--ifm-global-shadow-lw)",
        borderBottomLeftRadius: "var(--ifm-code-border-radius)",
        borderBottomRightRadius: "var(--ifm-code-border-radius)",
      }}
    >
      <h4 style={{ paddingLeft: "0.65rem", marginBottom: "0.45rem" }}>
        API Reference:
      </h4>
      <ul style={{ paddingBottom: "1rem" }}>
        {imports.map(({ imported, source, docs }) => (
          <li key={imported}>
            <a href={docs}>
              <span>{imported}</span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function CodeBlockWrapper({ children, ...props }) {
  // Initialize imports as an empty array
  let imports = [];

  // Check if children is a string
  if (typeof children === "string") {
    // Search for an IMPORTS comment in the code
    const match = /<!--IMPORTS:(.*?)-->\n/.exec(children);
    if (match) {
      imports = JSON.parse(match[1]);
      children = children.replace(match[0], "");
    }
  } else if (children.imports) {
    imports = children.imports;
  }

  return (
    <>
      <CodeBlock {...props}>{children}</CodeBlock>
      {imports.length > 0 && <Imports imports={imports} />}
    </>
  );
}