import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Lesson notes and the data script are read at build time from /content;
  // nothing in /content is served as a file.
  outputFileTracingIncludes: { "/**": ["./content/**/*"] },
};

export default nextConfig;
