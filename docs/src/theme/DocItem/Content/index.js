import React from "react";
import Content from "@theme-original/DocItem/Content";
import Feedback from "../../Feedback";
import LastUpdatedComponent from "../../LastUpdatedComponent";

export default function ContentWrapper(props) {
  return (
    <>
      {/* eslint-disable react/jsx-props-no-spreading */}
      <Content {...props} />
      <Feedback />
      <LastUpdatedComponent />
    </>
  );
}
