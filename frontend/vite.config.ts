/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backend = "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../src/calico/web/static/dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": backend,
      "/thumbs": backend,
      "^/runs/[^/]+/(image|reference)": backend,
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
