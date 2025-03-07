import { Fragment } from "react";
import MDXComponents from "@theme-original/MDXComponents";
import ChatModelTabs from "./ChatModelTabs.js"
import CodeBlock from "./CodeBlock";
import Compatibility from "./Compatibility.js";
import Div from "./Div.js";
import EmbeddingTabs from "./EmbeddingTabs.js";
import { ItemTable } from "./FeatureTables.js";
import Prerequisites from "./Prerequisites.js";
import Tabs from "@theme-original/Tabs";
import TabItem from "@theme-original/TabItem";
import VectorStoreTabs from "./VectorStoreTabs.js";

export default {
  ...MDXComponents,
  // Add your custom components here
  ChatModelTabs,
  CodeBlock,
  Compatibility,
  Div,
  EmbeddingTabs,
  Fragment,
  ItemTable,
  Prerequisites,
  TabItem,
  Tabs,
  VectorStoreTabs,
};
