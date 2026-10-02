// ui/vite.config.js  ── pure ESM
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { apiBase } from './src/api-config.js';

export default defineConfig(({ command, mode }) => ({
  plugins: [react()],
  define: {
    'import.meta.env.VITE_API_BASE': JSON.stringify(apiBase(loadEnv(mode, process.cwd()).VITE_API_BASE, {
      development: command === 'serve', render: mode === 'render',
    })),
  },
  server: {
    proxy: {
      "/v1": "http://localhost:8000",   // dev-time API proxy
    },
  },
}));
