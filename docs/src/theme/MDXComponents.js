import { Fragment } from "react";
import MDXComponents from "@theme-original/MDXComponents";
import ChatModelTabs from "./ChatModelTabs.js"
import CodeBlock from "./CodeBlock";
import Compatibility from "./Compatibility.js";
import DocPaginator from "./DocPaginator";
import Div from "./Div.js";
import EmbeddingTabs from "./EmbeddingTabs.js";
import FeatureTables from "./FeatureTables.js";
import People from "./People.js";
import Prerequisites from "./Prerequisites.js";
import SearchBar from "./SearchBar";
import SearchPage from "./SearchPage";
import Tabs from "@theme-original/Tabs";
import TabItem from "@theme-original/TabItem";
import VectorStoreTabs from "./VectorStoreTabs.js";

export default {
  ...MDXComponents,
  // Add your custom components here
  ChatModelTabs,
  CodeBlock,
  Compatibility,
  DocPaginator,
  Div,
  EmbeddingTabs,
  FeatureTables,
  Fragment,
  People,
  Prerequisites,
  SearchBar,
  SearchPage,
  TabItem,
  Tabs,
  VectorStoreTabs,
};
