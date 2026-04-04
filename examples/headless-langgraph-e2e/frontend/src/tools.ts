/**
 * Headless tool definitions + client implementations.
 *
 * Mirrors `agent.py`: schema-only headless tools; execution runs here via
 * `useStream({ tools: [...] })`.
 *
 * Pattern: https://github.com/langchain-ai/langchainjs (headless `tool` + `.implement()`)
 */
import { tool } from "langchain";
import { z } from "zod";

/** Must match `browser_navigate` in `agent.py`. */
export const browserNavigate = tool({
  name: "browser_navigate",
  description:
    "Navigate a headless browser to a URL. The real browser runs out-of-process; " +
    "the graph pauses until the client resumes with a result.",
  schema: z.object({
    url: z.string().describe("URL to open in the headless browser."),
  }),
});

/**
 * Client-side implementation: try `fetch` for same-origin or CORS-friendly URLs;
 * otherwise return a simulated result (many sites block browser cross-origin fetch).
 */
/** Must match `geolocation_get` in `agent.py`. */
export const geolocationGet = tool({
  name: "geolocation_get",
  description:
    "Get the user's current GPS coordinates using the browser's Geolocation API. " +
    "The client can show the position on OpenStreetMap.",
  schema: z.object({
    high_accuracy: z
      .boolean()
      .optional()
      .describe("Request high-accuracy GPS when supported."),
  }),
});

export const geolocationGetClient = geolocationGet.implement(async ({ high_accuracy }) => {
  if (!navigator.geolocation) {
    return JSON.stringify({
      success: false,
      message: "Geolocation is not supported by this browser.",
    });
  }

  const position = await new Promise<GeolocationPosition>((resolve, reject) => {
    navigator.geolocation.getCurrentPosition(resolve, reject, {
      enableHighAccuracy: high_accuracy ?? true,
      timeout: 10_000,
      maximumAge: 5 * 60 * 1000,
    });
  });

  const { latitude, longitude, accuracy } = position.coords;
  const timestamp = new Date(position.timestamp).toISOString();

  return JSON.stringify({
    success: true,
    latitude,
    longitude,
    accuracy,
    timestamp,
    message: `Location: ${latitude.toFixed(5)}, ${longitude.toFixed(5)} (±${Math.round(accuracy)} m)`,
  });
});

export const browserNavigateClient = browserNavigate.implement(async ({ url }) => {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return JSON.stringify({ ok: false, error: "Invalid URL", url });
  }

  try {
    const res = await fetch(parsed.href, { mode: "cors", credentials: "omit" });
    const text = await res.text();
    const titleMatch = text.match(/<title>([^<]*)<\/title>/i);
    return JSON.stringify({
      ok: true,
      url: parsed.href,
      status: res.status,
      title: titleMatch?.[1]?.trim() ?? null,
    });
  } catch {
    return JSON.stringify({
      ok: true,
      url: parsed.href,
      simulated: true,
      note:
        "Could not fetch page (often cross-origin). Simulated successful navigation for E2E.",
    });
  }
});
