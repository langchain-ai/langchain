import React from "react";
import { CssVarsProvider, getInitColorSchemeScript } from "@mui/joy/styles";
import CssBaseline from "@mui/joy/CssBaseline";

export default function Root({ children }) {
  return (
    <>
      {getInitColorSchemeScript()}
      <CssBaseline />
      <CssVarsProvider
        defaultMode="system"
        modeStorageKey="langsmith-docs-joy-mode"
      >
        {children}
      </CssVarsProvider>
    </>
  );
}
