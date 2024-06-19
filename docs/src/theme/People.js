import React from "react";
import PeopleData from "../../data/people.yml"

function renderPerson({ login, avatarUrl, url }) {
    return (
        <div key={`person:${login}`} style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "18px" }}>
            <a href={url} target="_blank">
                <img src={avatarUrl} style={{ borderRadius: "50%", width: "128px", height: "128px" }} />
            </a>
            <a href={url} target="_blank" style={{ fontSize: "18px", fontWeight: "700" }}>@{login}</a>
        </div>
    );
}

export default function People({ type, count }) {
    let people = PeopleData[type] ?? [];
    if (count !== undefined) {
      people = people.slice(0, parseInt(count, 10));
    }
    const html = people.map((person) => {
        return renderPerson(person);
    });
    return (
        <div style={{ display: "flex", flexWrap: "wrap", padding: "10px", justifyContent: "space-around" }}>
            {html}
        </div>
    );
}
