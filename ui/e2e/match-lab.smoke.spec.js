import { test, expect } from '@playwright/test';
for (const width of [1440, 820, 375, 320]) test(`Match Lab responsive keyboard/replay smoke at ${width}px`, async ({ page }, info) => {
  await page.setViewportSize({ width, height: 1000 }); await page.emulateMedia({ reducedMotion: 'reduce' });
  const tab = process.platform === 'darwin' && info.project.name === 'webkit' ? 'Alt+Tab' : 'Tab';
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/connect4/match-lab');
  const first = page.locator('input[name="player-1"][value="human"]');
  await first.focus(); await page.keyboard.press('Space'); await expect(first).toBeChecked();
  await page.locator('input[name="player-2"][value="random"]').check();
  await page.locator('#match-start').focus(); await page.keyboard.press('Enter');
  await expect(page.locator('#match-start')).toBeEnabled();
  const cell = page.getByRole('button', { name: 'Column 4, row 6: empty', exact: true });
  await page.locator('.cell[data-column="2"]').last().focus(); await page.keyboard.press(tab);
  await expect(cell).toBeFocused(); expect(await cell.evaluate(n => getComputedStyle(n).outlineStyle)).toBe('solid');
  await page.keyboard.press('Space'); await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('#match-next')).toBeEnabled();
  await page.locator('#match-next').focus(); await page.keyboard.press('Enter'); await expect(page.locator('.board-revision')).toHaveText('Move 2'); await expect(page.locator('#match-start')).toBeEnabled();
  await page.locator('#match-previous').focus(); await page.keyboard.press('Enter');
  await expect(page.locator('.board-revision')).toHaveText('Move 1'); await expect(page.locator('.cell:enabled')).toHaveCount(0);
  await page.locator('#match-forward').focus(); await page.keyboard.press('Enter'); await expect(page.locator('#match-view-mode')).toHaveText('LIVE');
  for (const node of ['#match-start', '#match-speed', '#match-previous', '.move-list button']) {
    const control = page.locator(node).first(); await control.focus();
    // Keyboard input makes the shared focus ring visible in every engine.
    await page.keyboard.press(tab); await page.keyboard.press(`Shift+${tab}`);
    expect(await control.evaluate(n => getComputedStyle(n).outlineStyle)).toBe('solid');
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(await page.locator('.move-list button').first().evaluate(n => n.getBoundingClientRect().height)).toBeGreaterThanOrEqual(44);
  const board = await page.locator('#game-board').boundingBox(); expect(Math.abs(board.width / board.height - 7 / 6)).toBeLessThan(.02);
  await page.screenshot({ path: info.outputPath(`match-lab-${width}.png`), fullPage: true }); expect(errors).toEqual([]);
});
