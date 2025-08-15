import React from "react";
import Admonition from '@theme/Admonition';

export default function Compatibility({ packagesAndVersions }) {
    return (
        <Admonition type="caution" title="Compatibility" icon="📦">
            <span style={{fontSize: "15px"}}>
              The code in this guide requires{" "}
              {packagesAndVersions.map(([pkg, version], i) => {
                return (
                  <span key={`compatibility-map${pkg}>=${version}-${i}`}>
                    <code>{`${pkg}>=${version}`}</code>
                    {i < packagesAndVersions.length - 1 && ", "}
                  </span>
                );
              })}.
              Please ensure you have the correct packages installed.
            </span>
        </Admonition>
    );
}