import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  // Vercel Services owns production routing. This proxy is for standalone local dev.
  async rewrites() {
    if (process.env.VERCEL) return [];
    const backend = process.env.BACKEND_URL || "http://127.0.0.1:8000";
    return [
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
      ...["/docs", "/openapi.json", "/redoc"].map((path) => ({ source: path, destination: `${backend}${path}` })),
    ];
  },
};
export default nextConfig;
