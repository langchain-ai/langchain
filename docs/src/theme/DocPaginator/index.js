import React from "react";
import DocPaginator from "@theme-original/DocPaginator";

const BLACKLISTED_PATHS = ["/docs/how_to/", "/docs/tutorials/"];

export default function DocPaginatorWrapper(props) {
  const [shouldHide, setShouldHide] = React.useState(false);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const currentPath = window.location.pathname;
    if (BLACKLISTED_PATHS.some((path) => currentPath.includes(path))) {
      setShouldHide(true);
    }
  }, []);

  if (!shouldHide) {
    // eslint-disable-next-line react/jsx-props-no-spreading
    return <DocPaginator {...props} />;
  }
  return null;
}
