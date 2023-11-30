export function ColumnContainer({children}) {
    return (
        <div style={{ clear: "both" }}>
            {children}
        </div>
    )
}

export function Column({children}) {
    return (
        <div style={{ width: "50%", float: "left", clear: "left", padding: "1%" }}>
            {children}
        </div>
    )
}