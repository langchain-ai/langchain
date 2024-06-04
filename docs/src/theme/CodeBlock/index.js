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
      <b style={{ paddingLeft: "0.65rem", marginBottom: "0.45rem", marginRight: "0.5rem" }}>
        API Reference:
      </b>
        {imports.map(({ imported, source, docs }, index) => (
          <span key={imported}>
            <a href={docs}>{imported}</a>{index < imports.length - 1 ? ' | ' : ''}
          </span>
        ))}
    </div>
  );
}

export default function CodeBlockWrapper({ children, ...props }) {
  if (typeof children === "string") {
    return <CodeBlock {...props}>{children}</CodeBlock>;
  }

  return (
    <>
      <CodeBlock {...props}>{children.content}</CodeBlock>
      <Imports imports={children.imports} />
    </>
  );
}
