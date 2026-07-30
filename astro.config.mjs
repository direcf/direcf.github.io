import { defineConfig } from "astro/config";

import sitemap from "@astrojs/sitemap";
import { readdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const SITE = "https://direcf.github.io";

// Static coursework ships as prebuilt HTML under public/posts/<slug>/ and is NOT
// an Astro route, so @astrojs/sitemap can't discover it — its pages were missing
// from the deployed sitemap entirely. Enumerate them here as customPages.
// Astro-native courses (src/data/courses/*.json) are skipped: they're already
// routed and sitemapped, and only keep an asset-only dir under public/posts/.
const postsDir = fileURLToPath(new URL("./public/posts", import.meta.url));
const coursesDir = fileURLToPath(new URL("./src/data/courses", import.meta.url));

function staticCoursePages() {
  const urls = [];
  for (const entry of readdirSync(postsDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const slug = entry.name;
    if (existsSync(`${coursesDir}/${slug}.json`)) continue; // Astro-native
    const dir = `${postsDir}/${slug}`;
    if (!existsSync(`${dir}/index.html`)) continue;
    const chapters = readdirSync(dir)
      .filter((f) => /^chapter-\d+\.html$/.test(f))
      .sort();
    if (chapters.length === 0) continue;
    urls.push(`${SITE}/posts/${slug}/`);
    for (const ch of chapters) urls.push(`${SITE}/posts/${slug}/${ch}`);
  }
  return urls;
}

export default defineConfig({
  site: SITE,

  // base: "/", // GitHub user-pages root, no base prefix
  build: {
    format: "directory", // /category/diary/ instead of /category/diary.html
  },

  integrations: [sitemap({ customPages: staticCoursePages() })],
});
