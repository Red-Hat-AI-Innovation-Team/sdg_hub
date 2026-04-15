import type { NextConfig } from "next";

const isGhPages = process.env.GITHUB_PAGES === "true";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  // For GitHub Pages: repo deploys to https://<org>.github.io/<repo>/
  basePath: isGhPages ? "/sdg_hub" : "",
  assetPrefix: isGhPages ? "/sdg_hub/" : "",
  trailingSlash: true,
};

export default nextConfig;
