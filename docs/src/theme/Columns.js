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
        <div style={{ flex: "1 0 300px", padding: "10px", overflowX: "scroll", zoom: '80%' }}>
            {children}
        </div>
    )
}
