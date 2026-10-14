import { test, expect } from '@playwright/test';
import { status } from '../tests/fixtures/alphazero.js';

// Run once with and once without the Victor flag; AlphaZero availability is mocked here.
const victor = (process.env.VICTOR_LABS_UI ?? process.env.VICTOR_RELEASE_UI) === 'true';
const near = (a, b) => Math.abs(a - b) <= 1;

const measure = page => page.locator('.competitor-choices').evaluateAll(grids => grids.map(grid => {
  const g = grid.getBoundingClientRect(), gap = parseFloat(getComputedStyle(grid).columnGap);
  const columns = getComputedStyle(grid).gridTemplateColumns.split(' ').length;
  return { width: g.width, gap, columns, options: [...grid.children].map(option => {
    const r = option.getBoundingClientRect(), label = option.querySelector('span');
    return { type: option.querySelector('input').value, left: r.left - g.left, right: g.right - r.right, top: r.top - g.top,
      width: r.width, height: r.height, lines: label.getClientRects().length, clipped: option.scrollWidth > option.clientWidth };
  }) };
}));

for (const alphaZero of [true, false]) for (const width of [1440, 1181, 1180, 820, 375, 320]) {
  test(`Match Lab competitor grid at ${width}px, AlphaZero ${alphaZero ? 'available' : 'absent'}, Victor ${victor ? 'on' : 'off'}`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    page.on('pageerror', error => { throw error; });
    // Discovery only: no game is started.
    await page.route('**/v1/**', route => route.request().url().endsWith('/agents/alphazero')
      ? route.fulfill(alphaZero ? { json: status } : { status: 404, json: { detail: 'Not found' } }) : route.abort());
    await page.goto('/connect4/match-lab');
    if (alphaZero) await expect(page.locator('input[name="player-2"][value="alphazero"]')).toBeVisible();
    else await expect(page.locator('input[name="player-2"][value="mcts_tactical"]')).toBeVisible();

    const order = ['human', 'random', 'negamax', 'mcts', 'mcts_tactical', ...(alphaZero ? ['alphazero'] : []), ...(victor ? ['victor_research'] : [])];
    const grids = await measure(page);
    expect(grids).toHaveLength(2);
    for (const grid of grids) {
      const { options, columns } = grid, cell = (grid.width - (columns - 1) * grid.gap) / columns;
      expect(options.map(o => o.type)).toEqual(order);
      expect(columns).toBe(width > 1180 ? 4 : 2);
      // Equal row heights, no clipped or overflowing labels, no awkward three-line wrapping.
      for (const o of options) {
        expect(near(o.height, options[0].height)).toBe(true);
        expect(o.clipped).toBe(false);
        expect(o.left).toBeGreaterThanOrEqual(-0.5); expect(o.right).toBeGreaterThanOrEqual(-0.5);
        expect(o.lines).toBeLessThanOrEqual(2);
      }
      const rows = new Set(options.map(o => Math.round(o.top))).size;
      const research = options.find(o => o.type === 'victor_research');
      if (columns === 4) {
        expect(rows).toBe(2);
        if (research) {
          // Victor shares the second row and spans two columns; with AlphaZero it fills the two rightmost.
          expect(near(research.top, options.find(o => o.type === 'mcts_tactical').top)).toBe(true);
          expect(near(research.width, 2 * cell + grid.gap)).toBe(true);
          if (alphaZero) expect(near(research.right, 0)).toBe(true);
          expect(research.lines).toBe(1);
        }
      } else {
        expect(rows).toBe(Math.ceil((order.length - (research ? 1 : 0)) / 2) + (research ? 1 : 0));
        if (research) expect(near(research.width, grid.width)).toBe(true);
      }
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });
}
