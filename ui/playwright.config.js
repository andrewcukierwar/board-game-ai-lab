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
  projects: [
    { name: 'chromium', use: { browserName: 'chromium' } },
    { name: 'firefox', testMatch: '**/polish.spec.js', use: { browserName: 'firefox' } },
    { name: 'webkit', testMatch: '**/polish.spec.js', use: { browserName: 'webkit' } },
  ],
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:3000',
    trace: 'retain-on-failure',
  },
});
