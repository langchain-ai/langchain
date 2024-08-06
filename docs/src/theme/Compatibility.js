import React from "react";
import Admonition from '@theme/Admonition';

export default function Compatibility({ packagesAndVersions }) {
    return (
        <Admonition type="caution" title="Compatibility" icon="📦">
            <span style={{fontSize: "15px"}}>
              The code in this guide requires{" "}
              {packagesAndVersions.map(([pkg, version], i) => {
                return (
                  <code key={`compatiblity-map${pkg}>=${version}-${i}`}>{`${pkg}>=${version}`}</code>
                );
              })}.
              Please ensure you have the correct packages installed.
            </span>
        </Admonition>
    );
}
