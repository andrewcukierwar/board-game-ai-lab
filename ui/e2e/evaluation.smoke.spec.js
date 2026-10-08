import { test, expect } from '@playwright/test';
import { readFile, mkdir } from 'node:fs/promises';
import { seedSeason, seasonFixture } from './fixtures/season.js';
import { verifyEvaluationExport } from '../src/evaluation/verify.js';
import { nodeSHA256 } from '../scripts/evaluation-node.mjs';
import { evaluationSeason } from '../tests/fixtures/evaluation.js';

for (const width of [1440, 375, 320]) test(`Evaluation export JSON, CSV, provenance, partial and complete at ${width}px`, async ({ page }, info) => {
  await page.setViewportSize({ width, height: 1000 });
  await page.addInitScript(() => {
    window.copiedDigest = null;
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async text => { window.copiedDigest = text; } } });
  });
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  await seedSeason(page, evaluationSeason({ size: 8, games: 4, count: 37 })); await page.goto('/connect4/season');
  const panel = page.getByRole('region', { name: 'Evaluation export' });
  await expect(panel).toContainText('Partial evaluation — 37 of 112 games complete.');
  await expect(panel.getByRole('button', { name: 'Export evaluation JSON' })).toBeEnabled();
  await expect(panel.locator('.evidence-digest')).toHaveText(/^[0-9a-f]{64}$/);
  const digest = await panel.locator('.evidence-digest').textContent();
  await panel.getByRole('button', { name: 'Copy evidence digest' }).focus(); await page.keyboard.press('Enter');
  await expect(panel.getByRole('status')).toHaveText('Evidence digest copied.'); expect(await page.evaluate(() => window.copiedDigest)).toBe(digest);
  for (const [label, suffix, rows] of [['Export evaluation JSON', 'evaluation.json', null], ['Export games CSV', 'games.csv', 38], ['Export summary CSV', 'summary.csv', 9]]) {
    const event = page.waitForEvent('download'); await panel.getByRole('button', { name: label, exact: true }).click(); const download = await event;
    expect(download.suggestedFilename()).toBe(`connect4-season-1234-${suffix}`);
    const content = await readFile(await download.path(), 'utf8');
    if (rows) expect(content.trim().split('\r\n')).toHaveLength(rows);
    else { const artifact = JSON.parse(content); expect((await verifyEvaluationExport(artifact, { digest: nodeSHA256 })).ok).toBe(true); expect(artifact.integrity.evidence_sha256).toBe(digest); }
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await panel.scrollIntoViewIfNeeded();
  if (info.project.name === 'chromium') {
    await mkdir('playwright-report/phase5e', { recursive: true });
    await panel.screenshot({ path: `playwright-report/phase5e/partial-${width}.png` });
  }
  await panel.getByText('Provenance and integrity details', { exact: true }).click(); await expect(panel).toContainText('corrected Negamax 2');
  if (info.project.name === 'chromium' && width === 1440) await panel.screenshot({ path: 'playwright-report/phase5e/provenance-integrity.png' });
  await seedSeason(page, evaluationSeason({ size: 8, games: 4 })); await page.reload();
  await expect(panel).toContainText('Complete evaluation — 112 of 112 games complete.');
  await expect(panel.getByRole('button', { name: 'Export evaluation JSON' })).toBeEnabled();
  const event = page.waitForEvent('download'); await panel.getByRole('button', { name: 'Export evaluation JSON', exact: true }).click();
  const complete = JSON.parse(await readFile(await (await event).path(), 'utf8'));
  expect(complete.evidence.state.status).toBe('complete'); expect((await verifyEvaluationExport(complete, { digest: nodeSHA256 })).ok).toBe(true);
  const button = panel.getByRole('button', { name: 'Export games CSV', exact: true }); await button.focus();
  const tab = info.project.name === 'webkit' && process.platform === 'darwin' ? 'Alt+Tab' : 'Tab';
  await page.keyboard.press(tab); await page.keyboard.press(`Shift+${tab}`); await expect(button).toBeFocused();
  expect(await button.evaluate(n => getComputedStyle(n).outlineStyle)).toBe('solid');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  if (info.project.name === 'chromium') {
    await panel.screenshot({ path: `playwright-report/phase5e/complete-${width}.png` });
    if (width === 1440) { await page.evaluate(() => window.scrollTo(0, 0)); await page.screenshot({ path: 'playwright-report/phase5e/desktop-complete.png', fullPage: true }); }
  }
  expect(errors).toEqual([]);
});
test('Web Crypto failure blocks downloads with an actionable error', async ({ page }) => {
  await seedSeason(page, seasonFixture(4, 2, 1));
  await page.addInitScript(() => Object.defineProperty(crypto, 'subtle', { value: {}, configurable: true }));
  await page.goto('/connect4/season'); const panel = page.getByRole('region', { name: 'Evaluation export' });
  await expect(panel.getByRole('alert')).toContainText('Web Crypto is unavailable');
  await expect(panel.getByRole('button', { name: 'Export evaluation JSON' })).toBeDisabled();
});
