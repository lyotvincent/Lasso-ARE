import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [
    react(),
    {
      name: "ascii-calendar-regex",
      apply: "build",
      // esbuild's charset option escapes strings but leaves regex literals intact.
      // Preserve Plotly's calendar regex semantics using equivalent ASCII escapes.
      renderChunk: {
        order: "post",
        handler(code) {
          return {
            code: code.replace(/[\u3400-\u9fff\uf900-\ufaff]/g, (char) =>
              "\\u" + char.charCodeAt(0).toString(16).padStart(4, "0")
            ),
            map: null,
          };
        },
      },
    },
  ],
  // Keep bundled dependency strings safe for ASCII-only source viewers.
  esbuild: {
    charset: "ascii",
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
