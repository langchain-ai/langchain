import * as path from "path"

import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// The dev server proxies `/api/*` to the FastAPI backend (port 8000 by default),
// so the frontend can call relative URLs and avoid CORS during development. Set
// VITE_API_TARGET to point at a backend on a different host/port.
const apiTarget = process.env.VITE_API_TARGET ?? "http://localhost:8000"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": apiTarget,
    },
  },
})
