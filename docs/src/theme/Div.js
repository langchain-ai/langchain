import React from 'react';

export default function Div({ value, ...props }) {
    return <div {...props}>{value}</div>;
}
