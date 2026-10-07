import { test, expect } from '@playwright/test';

const repository = 'https://github.com/andrewcukierwar/board-game-ai-lab';

for (const [name, width, height] of [['desktop', 1440, 1000], ['tablet', 820, 1100], ['mobile', 375, 812], ['narrow mobile', 320, 740]]) {
  test(`homepage and game navigation work on ${name}`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height });
    const errors = [];
    const apiRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    // Shell navigation should never initialize a game or request an explanation.
    await page.route('**/v1/**', route => {
      apiRequests.push(route.request().url());
      return route.abort();
    });
    await page.goto('/');
    await expect(page).toHaveTitle('Board Game AI Lab');
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('Board GameAI Lab.');
    await expect(page.getByText('Explore how game-playing AI chooses moves.')).toBeVisible();
    for (const agent of ['Random', 'Negamax', 'MCTS']) {
      await expect(page.locator('#agents').getByRole('heading', { name: agent, exact: true })).toBeVisible();
    }
    await expect(page.locator('#research')).toContainText('is not available in the public game');
    const navigation = page.getByRole('navigation', { name: 'Main navigation' });
    for (const link of ['Play', 'Agents', 'Research', 'GitHub']) {
      await expect(navigation.getByRole('link', { name: link })).toBeVisible();
    }
    await expect(navigation.getByRole('link', { name: 'GitHub' })).toHaveAttribute('href', repository);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`home-${width}.png`), fullPage: true });
    await page.screenshot({ path: testInfo.outputPath(`hero-${width}.png`) });

    await page.getByRole('link', { name: 'Play Connect 4', exact: true }).click();
    await expect(page).toHaveURL(/\/connect4$/);
    await expect(page.getByRole('heading', { level: 1, name: 'Connect 4' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Start game', exact: true })).toBeVisible();
    await expect(page.locator('input[name="opponent"][value="negamax"]')).toBeChecked();
    await expect(page.getByLabel('Search depth:')).toBeVisible();
    await expect(page.locator('#analyze-position')).toBeDisabled();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.getByRole('link', { name: 'Home', exact: true }).click();
    await expect(page).toHaveURL(/\/$/);
    await navigation.getByRole('link', { name: 'Research' }).click();
    await expect(page).toHaveURL(/\/#research$/);
    await expect(page.locator('#research')).toBeFocused();
    await navigation.getByRole('link', { name: 'Play', exact: true }).click();
    await navigation.getByRole('link', { name: 'Agents', exact: true }).click();
    await expect(page).toHaveURL(/\/#agents$/);
    await expect(page.locator('#agents')).toBeFocused();
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await navigation.getByRole('link', { name: 'Agents', exact: true }).click();
    await expect.poll(() => page.locator('#agents').evaluate(section => Math.abs(section.getBoundingClientRect().top - 24))).toBeLessThan(2);
    expect(apiRequests).toEqual([]);
    expect(errors).toEqual([]);
  });
}

test('keyboard skip link and brand support navigation after a direct game refresh', async ({ page }) => {
  await page.goto('/connect4');
  await page.reload();
  await expect(page.getByRole('button', { name: 'Start game', exact: true })).toBeVisible();
  const skip = page.getByRole('link', { name: 'Skip to content' });
  expect(await skip.evaluate(node => node.getBoundingClientRect().bottom)).toBeLessThanOrEqual(0);
  await page.keyboard.press('Tab');
  await expect(skip).toBeFocused();
  await expect(skip).toBeInViewport();
  const bounds = await skip.boundingBox();
  expect(bounds.x).toBe(12);
  expect(bounds.y).toBe(12);
  await page.keyboard.press('Enter');
  await expect(page.locator('#main-content')).toBeFocused();
  expect(await skip.evaluate(node => node.getBoundingClientRect().bottom)).toBeLessThanOrEqual(0);
  await page.getByRole('link', { name: 'Board Game AI Lab home', exact: true }).click();
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await expect(page.locator('meta[name="description"]')).toHaveAttribute('content', /grounded position analysis/);
});

for (const width of [820, 375, 320]) {
  test(`React board fits and supports keyboard play at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.route('**/v1/connect4/explain', route => route.abort());
    await page.goto('/connect4');
    await page.locator('input[name="opponent"][value="random"]').check();
    await page.getByRole('button', { name: 'Start game', exact: true }).click();
    await expect(page.locator('#message')).toContainText('Your turn');
    await expect(page.locator('#game-board .cell')).toHaveCount(42);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    expect(await page.locator('#game-board').evaluate(board => {
      const bounds = board.getBoundingClientRect();
      const surface = board.closest('main').getBoundingClientRect();
      return bounds.left >= surface.left && bounds.right <= surface.right;
    })).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`game-${width}.png`), fullPage: true });
    const cell = page.locator('.cell[data-column="3"]:enabled').first();
    await cell.focus();
    await page.keyboard.press('Enter');
    await expect(page.locator('.circle.x')).toHaveCount(1);
    await expect(page.locator('.circle.o')).toHaveCount(1);
    await expect(page.locator('#message')).toContainText('Your turn');
    await page.getByRole('link', { name: 'Board Game AI Lab home', exact: true }).click();
    await expect(page.getByRole('link', { name: 'Play Connect 4', exact: true })).toBeVisible();
    expect(errors).toEqual([]);
  });
}
