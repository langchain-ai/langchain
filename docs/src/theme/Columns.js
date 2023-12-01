import React from "react";

export function ColumnContainer({children}) {
    return (
        <div style={{ display: "flex", flexWrap: "wrap" }}>
            {children}
        </div>
    )
}

export function Column({children}) {
    return (
        <div style={{ flex: "1 1 50%", padding: "1%" }}>
            {children}
        </div>
    )
}
