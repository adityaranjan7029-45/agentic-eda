import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  output: "standalone", // small self-contained server for the Docker image (step 15)
};

export default nextConfig;
