import { defineConfig } from '@playwright/test';

for (const value of [process.env.PLAYWRIGHT_BASE_URL, process.env.PLAYWRIGHT_API_URL]) {
  if (value && !['localhost', '127.0.0.1', '[::1]'].includes(new URL(value).hostname)) {
    throw new Error('Gameplay regression tests must target local servers, never production.');
  }
}

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 60000,
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:3000',
    browserName: 'chromium',
    trace: 'retain-on-failure',
  },
});
