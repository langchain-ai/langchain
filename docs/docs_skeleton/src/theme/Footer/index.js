import React from "react";
import Footer from "@theme-original/Footer";
import Docsly from "@docsly/react";
import "@docsly/react/styles.css";
import { useLocation } from "@docusaurus/router";
 
export default function FooterWrapper(props) {
  const { pathname } = useLocation();
  return (
    <>
      <Footer {...props} />
      <Docsly publicId="public_FpkRFtHG3zlOAQ1PwmZ84zpg4bEt3od1jh6B648GpXJ4JrXvSEH8j5yazVwG07KQ" pathname={pathname} />
    </>
  );
}