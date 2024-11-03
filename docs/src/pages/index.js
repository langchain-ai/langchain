/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from "react";
import { Redirect } from "@docusaurus/router";
import useBaseUrl from "@docusaurus/useBaseUrl";

export default function Home() {
  return <Redirect to={useBaseUrl("/docs/introduction/")} />;
}
