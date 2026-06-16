import { defineConfig } from "astro/config";

import sitemap from "@astrojs/sitemap";

export default defineConfig({
  site: "https://direcf.github.io",

  // base: "/", // GitHub user-pages root, no base prefix
  build: {
    format: "directory", // /category/diary/ instead of /category/diary.html
  },

  integrations: [sitemap()],
});