/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */
import React from "react";
import { MendableSearchBar } from "@mendable/search";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

export default function SearchBarWrapper() {
  const {
    siteConfig: { customFields },
  } = useDocusaurusContext();
  return (
    <div className="mendable-search">
      <MendableSearchBar
        anon_key={customFields.mendableAnonKey}
        style={{ accentColor: "#4F956C", darkMode: false }}
        placeholder="Search..."
        dialogPlaceholder="How do I use a LLM Chain?"
        messageSettings={{ openSourcesInNewTab: false, prettySources: true }}
        showSimpleSearch
      />
    </div>
  );
}
