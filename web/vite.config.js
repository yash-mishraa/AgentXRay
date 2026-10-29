import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

export default {
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        quickstart: resolve(__dirname, "quickstart.html"),
        docs: resolve(__dirname, "docs.html"),
        playground: resolve(__dirname, "playground.html"),
        research: resolve(__dirname, "research.html"),
      },
    },
  },
};
