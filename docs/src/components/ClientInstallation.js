import React from "react";
import { CodeTabs } from "./InstructionsWithCode";

export function ClientInstallationCodeTabs() {
  return (
    <CodeTabs
      groupId="client-language"
      tabs={[
        {
          value: "python",
          label: "pip",
          language: "bash",
          content: `pip install -U langsmith`,
        },
        {
          value: "typescript",
          label: "yarn",
          language: "bash",
          content: `yarn add langsmith`,
        },
        {
          value: "npm",
          label: "npm",
          language: "bash",
          content: `npm install -S langsmith`,
        },
        {
          value: "pnpm",
          label: "pnpm",
          language: "bash",
          content: `pnpm add langsmith`,
        },
      ]}
    />
  );
}
