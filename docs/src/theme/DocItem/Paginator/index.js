import React from 'react';
import Paginator from '@theme-original/DocItem/Paginator';
import Feedback from "@theme/Feedback";

export default function PaginatorWrapper(props) {
  return (
    <>
      <Feedback />
      <Paginator {...props} />
    </>
  );
}
