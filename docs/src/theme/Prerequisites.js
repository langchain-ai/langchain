import React from "react";
import Admonition from '@theme/Admonition';

export default function Prerequisites({ titlesAndLinks }) {
    return (
        <Admonition type="info" title="Prerequisites" icon="📚">
            <ul style={{ fontSize: "15px", lineHeight: "1.5em" }}>
                {titlesAndLinks.map(([title, link], i) => {
                    return (
                        <li key={`prereq-${link.replace(/\//g, "")}-${i}`}>
                          <a href={link}>{title}</a>
                        </li>
                    );
                })}
            </ul>
        </Admonition>
    );
}
