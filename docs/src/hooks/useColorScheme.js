import { useColorScheme as useColorSchemeMui } from "@mui/joy/styles";

// Same theme logic as in smith-frontend
export function useColorScheme() {
  const { systemMode, setMode } = useColorSchemeMui();
  const isDarkMode = systemMode === "dark";

  return {
    mode: systemMode,
    isDarkMode,
    isLightMode: !isDarkMode,
    setMode,
  };
}
