import { test, expect } from '@playwright/test';

const fits = page => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
async function start(page) {
  await page.goto('/connect4');
  await page.locator('input[name="opponent"][value="random"]').check();
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.locator('#message')).toContainText('Your turn');
}

for (const [name, width, height] of [['desktop', 1440, 1100], ['tablet', 820, 1100], ['mobile', 375, 812], ['narrow-mobile', 320, 740]]) {
  test(`active workspace, keyboard focus and analysis placement at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await start(page);
    const cell = page.getByRole('button', { name: 'Column 4, row 6: empty', exact: true });
    await page.locator('.cell[data-column="2"]').last().focus();
    await page.keyboard.press('Tab');
    await expect(cell).toBeFocused();
    expect(await cell.evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: testInfo.outputPath(`connect4-${name}-focus.png`), fullPage: true });
    await page.keyboard.press('Space');
    await expect(page.locator('.circle.x')).toHaveCount(1);
    await expect(page.locator('.circle.o')).toHaveCount(1);
    await expect(page.locator('#message')).toContainText('Your turn');
    await expect(page.locator('#restart-button')).toBeEnabled();
    await expect(page.locator('#what-if-column')).toHaveCount(0);
    await expect(page.locator('#column-help')).toHaveCount(0);
    await page.evaluate(() => window.scrollTo(0, 0));
    expect(await fits(page)).toBe(true);
    const board = await page.locator('#game-board').boundingBox();
    const setup = await page.locator('.opponent-panel').boundingBox();
    const status = await page.locator('.game-status').boundingBox();
    const analysis = await page.locator('#explanation-panel').boundingBox();
    expect(Math.abs(board.width / board.height - 7 / 6)).toBeLessThan(.02);
    expect(board.x).toBeGreaterThanOrEqual(0);
    expect(board.x + board.width).toBeLessThanOrEqual(width);
    if (width > 760) expect(setup.x).toBeGreaterThan(board.x + board.width);
    else {
      expect(status.y + status.height).toBeLessThan(board.y);
      expect(setup.y).toBeGreaterThan(board.y + board.height);
    }
    expect(analysis.y).toBeGreaterThan(setup.y + setup.height);
    await page.screenshot({ path: testInfo.outputPath(`connect4-${name}-active.png`), fullPage: true });
    expect(errors).toEqual([]);
  });
}

test('AI thinking locks gameplay and uncertain recovery keeps long errors readable', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 320, height: 740 });
  await start(page);
  let release, entered;
  const gate = new Promise(resolve => { release = resolve; });
  const began = new Promise(resolve => { entered = resolve; });
  const error = 'The game server could not finish the request. Please check the authoritative board before continuing. '.repeat(5);
  await page.route('**/v1/connect4/make_move', async route => {
    if ('column' in route.request().postDataJSON()) return route.continue();
    entered();
    await gate;
    await route.fulfill({ status: 503, json: { error } });
  });
  await page.locator('.cell[data-column="3"]').first().click();
  await began;
  await expect(page.getByRole('status')).toContainText('AI thinking');
  await expect(page.locator('.cell:enabled')).toHaveCount(0);
  await expect(page.locator('#restart-button')).toBeDisabled();
  await expect(page.locator('#analyze-position')).toBeDisabled();
  await page.screenshot({ path: testInfo.outputPath('ai-thinking-320.png'), fullPage: true });
  // Losing both the POST response and reconciliation must require explicit GET.
  await page.route('**/v1/connect4/games/*', route => route.abort(), { times: 1 });
  release();
  await expect(page.getByRole('button', { name: 'Refresh game', exact: true })).toBeVisible();
  await expect(page.locator('.cell:enabled')).toHaveCount(0);
  await expect(page.locator('#message')).toContainText('Refresh the game before continuing');
  expect(await fits(page)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('uncertain-long-error-320.png'), fullPage: true });
  await page.unroute('**/v1/connect4/make_move');
  await page.getByRole('button', { name: 'Refresh game', exact: true }).click();
  await expect(page.locator('#message')).toContainText('Your turn');
  await expect(page.locator('.circle.x')).toHaveCount(1);
  await expect(page.locator('.circle.o')).toHaveCount(1);
});

test('navigation away during gameplay ignores a late start response', async ({ page }) => {
  let release, entered;
  const gate = new Promise(resolve => { release = resolve; });
  const began = new Promise(resolve => { entered = resolve; });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/v1/connect4/start_game', async route => {
    const response = await route.fetch();
    entered();
    await gate;
    try { await route.fulfill({ response }); } catch { /* aborted by navigation */ }
  });
  await page.goto('/connect4');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await began;
  await page.getByRole('link', { name: 'Home', exact: true }).click();
  release();
  await page.unroute('**/v1/connect4/start_game');
  await page.getByRole('link', { name: 'Play Connect 4', exact: true }).click();
  await expect(page.locator('#start-button')).toBeEnabled();
  await expect(page.locator('#restart-button')).toBeHidden();
  await expect(page.locator('#game-board .cell')).toHaveCount(0);
  expect(errors).toEqual([]);
});
