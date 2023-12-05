import React from "react";
import Footer from "@theme-original/Footer";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { MendableFloatingButton } from "@mendable/search";
import { useColorMode } from "@docusaurus/theme-common";

export default function FooterWrapper(props) {
  const {
    siteConfig: { customFields },
  } = useDocusaurusContext();

  const { colorMode, setColorMode } = useColorMode();

  return (
    <>
      <div style={{ zIndex: 1000 }}>
        <MendableFloatingButton
          anon_key={customFields.mendableAnonKey}
          style={{
            accentColor: "#4F956C",
            darkMode: colorMode === "dark" ? true : false,
          }}
          dialogPlaceholder="How do I use a LLM Chain?"
          messageSettings={{ openSourcesInNewTab: false, prettySources: true, sourcesFirst: true }}
          floatingButtonStyle={{
            backgroundColor: colorMode === "dark" ? "#4F956C" : "#1C1E21",
            color: "#FFFFFF",
          }}
          botIcon={<div style={{ fontSize: "22px" }}>🦜️</div>}
          icon={<div style={{ fontSize: "32px" }}>🦜️</div>}
          isPinnable
        />
      </div>
      <Footer {...props} />
    </>
  );
}
