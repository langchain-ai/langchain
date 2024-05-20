import React from "react";
import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";
import CodeBlock from "@theme/CodeBlock";
import { marked } from "marked";
import DOMPurify from "isomorphic-dompurify";
import prettier from "prettier";
import dedent from "dedent";
import parserTypeScript from "prettier/parser-typescript";

export function LangChainPyBlock(content) {
  return {
    value: "langchain-py",
    label: "LangChain (Python)",
    content,
    language: "python",
  };
}

export function LangChainJSBlock(content) {
  return {
    value: "langchain-js",
    label: "LangChain (JS)",
    content,
    language: "typescript",
  };
}

export function TypeScriptBlock(content, caption = "", label = "TypeScript") {
  return {
    value: "typescript",
    label,
    content,
    caption,
    language: "typescript",
  };
}

export function PythonBlock(content, caption = "", label = "Python") {
  return {
    value: "python",
    label,
    content,
    caption,
    language: "python",
  };
}

export function APIBlock(content, caption = "") {
  return {
    value: "api",
    label: "API (Using Python Requests)",
    content,
    caption,
    language: "python",
  };
}

export function ShellBlock(content, value = "shell", label = "Shell") {
  return {
    value,
    label,
    content,
    language: "shell",
  };
}

/**
 * @param {string} code
 * @param {"typescript" | "python"} language
 * @returns {string} The formatted code
 */
function formatCode(code, language) {
  const languageLower = language.toLowerCase();
  if (languageLower === "python") {
    // no-op - Do not format Python code at this time
    return code;
  }

  try {
    const formattedCode = prettier.format(code, {
      parser: languageLower,
      plugins: [parserTypeScript],
    });

    return formattedCode;
  } catch (_) {
    // no-op
  }

  // If formatting fails, return as is
  return code;
}

export function CodeTabs({ tabs, groupId }) {
  return (
    <Tabs groupId={groupId} className="code-tabs">
      {tabs.map((tab, index) => {
        const key = `${groupId}-${index}`;
        return (
          <TabItem key={key} value={tab.value} label={tab.label}>
            {tab.caption && (
              <div
                className="code-caption"
                // eslint-disable-next-line react/no-danger
                dangerouslySetInnerHTML={{
                  __html: DOMPurify.sanitize(marked.parse(tab.caption)),
                }}
              />
            )}
            <CodeBlock
              className={tab.value}
              language={tab.language ?? tab.value}
            >
              {formatCode(tab.content, tab.language ?? tab.value)}
            </CodeBlock>
          </TabItem>
        );
      })}
    </Tabs>
  );
}

export const typescript = (strings, ...values) => {
  if (
    values.length === 0 &&
    typeof strings === "object" &&
    strings != null &&
    !Array.isArray(strings)
  ) {
    const { caption, label } = strings;
    return (innerStrings, ...innerValues) => {
      let result = "";
      innerStrings.forEach((string, i) => {
        result += string + String(innerValues[i] ?? "");
      });
      return TypeScriptBlock(
        dedent(result),
        caption ?? undefined,
        label ?? undefined
      );
    };
  }

  let result = "";
  strings.forEach((string, i) => {
    result += string + String(values[i] ?? "");
  });
  return TypeScriptBlock(dedent(result));
};

export const python = (strings, ...values) => {
  if (
    values.length === 0 &&
    typeof strings === "object" &&
    strings != null &&
    !Array.isArray(strings)
  ) {
    const { caption, label } = strings;
    return (innerStrings, ...innerValues) => {
      let result = "";
      innerStrings.forEach((string, i) => {
        result += string + String(innerValues[i] ?? "");
      });
      return PythonBlock(
        dedent(result),
        caption ?? undefined,
        label ?? undefined
      );
    };
  }

  let result = "";
  strings.forEach((string, i) => {
    result += string + String(values[i] ?? "");
  });
  return PythonBlock(dedent(result));
};

export const shell = (strings, ...values) => {
  if (
    values.length === 0 &&
    typeof strings === "object" &&
    strings != null &&
    !Array.isArray(strings)
  ) {
    const { value, label } = strings;
    return (innerStrings, ...innerValues) => {
      let result = "";
      innerStrings.forEach((string, i) => {
        result += string + String(innerValues[i] ?? "");
      });
      return ShellBlock(dedent(result), value ?? undefined, label ?? undefined);
    };
  }

  let result = "";
  strings.forEach((string, i) => {
    result += string + String(values[i] ?? "");
  });
  return ShellBlock(dedent(result));
};
