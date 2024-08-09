import React, { useEffect } from "react";
import ColorModeToggle from "@theme-original/ColorModeToggle";
import { useColorScheme } from "../../hooks/useColorScheme";

export default function ColorModeToggleWrapper(props) {
  const { setMode } = useColorScheme();

  // "value" holds the color theme. Either "light" or "dark"
  const { value } = props;

  // change mode based on "value" prop
  // "dark" or "light" are also used for MUI
  useEffect(() => {
    setMode(value);
  }, [value]);

  return <ColorModeToggle {...props} />;
}
