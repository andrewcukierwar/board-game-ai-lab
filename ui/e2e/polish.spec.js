import { test, expect } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { representative } from './fixtures/analysis.js';

const fits = page => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth);
// Keep portfolio captures outside Playwright's disposable output directory.
const screenshots = resolve(process.env.UI_SCREENSHOT_DIR || 'playwright-report/portfolio');

for (const width of [1440, 1024, 820, 768, 375, 320]) {
  test(`final frontend smoke, accessibility and layout at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: width > 768 ? 1000 : 812 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    // macOS WebKit uses Option-Tab to include links in keyboard traversal.
    const tabKey = process.platform === 'darwin' && testInfo.project.name === 'webkit' ? 'Alt+Tab' : 'Tab';
    const settleFocus = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.route('**/v1/connect4/explain', route => {
      const data = representative(route.request().postDataJSON());
      // Presentation fixture fits the real game's guaranteed first red piece.
      data.explanation.summary.text = 'Your piece on d1 supports d2 immediately above it. Gravity determines which squares can be played; this reachability check does not prove a long-term win.';
      data.explanation.facts = [
        { id: 'position', text: 'Your red piece on d1 supports the square d2 immediately above it.', classification: 'confirmed_tactical' },
        { id: 'reply', text: 'Only legal columns can receive a piece. Geometric completion squares are not automatically playable.', classification: 'confirmed_tactical' },
      ];
      data.explanation.key_facts = data.explanation.facts;
      // Exercise citation wrapping with the curated winning-square references
      // recorded in games/connect4/grounding/knowledge.py.
      data.explanation.strategic_context[0].source.references = [
        { chapter: 3, section: '3.1', thesis_pages: [16, 18] },
        { chapter: 3, section: '3.2', thesis_pages: [18, 19] },
        { chapter: 5, section: '5', thesis_pages: [32, 32] },
      ];
      return route.fulfill({ json: data });
    });
    await mkdir(`${screenshots}/${testInfo.project.name}`, { recursive: true });
    const capture = async name => {
      await page.mouse.move(0, 0);
      await page.screenshot({ path: `${screenshots}/${testInfo.project.name}/${name}-${width}.png`, fullPage: true });
    };

    await page.goto('/');
    await expect(page).toHaveTitle('Board Game AI Lab');
    await expect(page.getByRole('main')).toHaveCount(1);
    await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1);
    await expect(page.locator('meta[property="og:title"]')).toHaveAttribute('content', 'Board Game AI Lab');
    await expect(page.locator('link[rel="icon"]')).toHaveAttribute('href', '/favicon.svg');
    const nav = page.getByRole('navigation', { name: 'Main navigation' });
    if (width <= 375) {
      const boxes = await nav.getByRole('link').evaluateAll(links => links.map(link => {
        const { y, height } = link.getBoundingClientRect(); return { y, height };
      }));
      expect(new Set(boxes.map(box => box.y)).size).toBe(1);
      expect(boxes.every(box => box.height >= 44)).toBe(true);
      expect((await page.locator('.site-header').boundingBox()).height).toBeLessThan(120);
    }
    expect(await fits(page)).toBe(true);
    await capture('homepage');
    const skip = page.getByRole('link', { name: 'Skip to content' });
    expect(await skip.evaluate(node => node.getBoundingClientRect().bottom)).toBeLessThanOrEqual(0);
    await page.keyboard.press(tabKey);
    await expect(skip).toBeFocused();
    await expect(skip).toBeInViewport();
    await page.keyboard.press('Enter');
    await expect(page.locator('#main-content')).toBeFocused();
    await settleFocus();
    expect(await page.locator('#main-content').evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
    expect(await skip.evaluate(node => node.getBoundingClientRect().bottom)).toBeLessThanOrEqual(0);

    // Keyboard focus on navigation and the hero CTA uses the shared link style.
    await nav.getByRole('link', { name: 'Play', exact: true }).focus();
    await page.keyboard.press(tabKey);
    await expect(nav.getByRole('link', { name: 'Agents', exact: true })).toBeFocused();
    expect(await nav.getByRole('link', { name: 'Agents', exact: true }).evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
    const cta = page.getByRole('link', { name: 'Play Connect 4', exact: true });
    await nav.getByRole('link', { name: 'GitHub' }).focus();
    await page.keyboard.press(tabKey);
    await expect(cta).toBeFocused();
    expect(await cta.evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL(/\/connect4$/);
    await page.reload(); // Direct route / SPA fallback.
    await settleFocus();
    await expect(page.getByRole('main')).toHaveCount(1);
    await expect(nav.getByRole('link', { name: 'Play', exact: true })).toHaveAttribute('aria-current', 'page');
    for (const agent of ['Random', 'Negamax', 'MCTS']) {
      await expect(page.getByRole('radio', { name: new RegExp(`^${agent}`) })).toBeVisible();
    }
    const random = page.getByRole('radio', { name: /^Random/ });
    await random.check();
    await expect(random).toBeChecked();
    await expect(page.locator('.agent-option.is-selected')).toContainText('Random');
    await random.focus();
    await page.keyboard.press(tabKey);
    await expect(page.getByRole('radio', { name: 'You go first' })).toBeFocused();
    await page.keyboard.press(tabKey);
    await expect(page.locator('#start-button')).toBeFocused();
    expect(await page.locator('#start-button').evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
    await page.keyboard.press(`Shift+${tabKey}`);
    await expect(page.getByRole('radio', { name: 'You go first' })).toBeFocused();
    await page.keyboard.press(`Shift+${tabKey}`);
    await expect(random).toBeFocused();
    expect(await page.locator('.agent-option.is-selected').evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
    await page.keyboard.press(tabKey);
    await page.keyboard.press(tabKey);
    await page.keyboard.press('Enter');
    await expect(page.locator('#message')).toContainText('Your turn');
    await expect(page.getByRole('group', { name: 'Connect 4 board' })).toHaveAttribute('aria-busy', 'false');
    await page.locator('.cell[data-column="2"]').last().focus();
    await page.keyboard.press(tabKey);
    const cell = page.getByRole('button', { name: 'Column 4, row 6: empty', exact: true });
    await expect(cell).toBeFocused();
    expect(await cell.evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
    await page.keyboard.press('Space');
    await expect(page.locator('.circle.x')).toHaveCount(1);
    await expect(page.locator('.circle.o')).toHaveCount(1);
    for (const column of [2, 4]) {
      await expect(page.locator('#message')).toContainText('Your turn');
      await page.locator(`.cell[data-column="${column}"]:enabled`).last().click();
      await expect(page.locator('#loading')).toBeHidden();
    }
    await expect(page.locator('.circle.x')).toHaveCount(3);
    await expect(page.locator('.circle.o')).toHaveCount(3);
    expect(await fits(page)).toBe(true);
    await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); });
    await capture('connect4-active');

    await expect(page.locator('#explanation-question')).toHaveAccessibleName('Optional question');
    await expect(page.locator('#question-count')).toHaveText('0/500');
    await page.locator('#explanation-question').fill('Which squares are reachable under gravity?');
    await page.locator('#what-if').click();
    await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
    await expect(page.locator('#what-if')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByLabel('Hypothetical move:')).toBeVisible();
    await page.locator('#what-if').focus();
    await page.keyboard.press(tabKey);
    await expect(page.locator('#explanation-question')).toBeFocused();
    expect(await page.locator('#explanation-question').evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
    await page.keyboard.press(tabKey);
    await expect(page.locator('#what-if-column')).toBeFocused();
    expect(await page.locator('#what-if-column').evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
    expect(await fits(page)).toBe(true);
    await page.locator('#analyze-position').click();
    await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
    await expect(page.locator('#analyze-position')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#what-if')).toHaveAttribute('aria-pressed', 'false');
    await expect(page.locator('#what-if-column')).toHaveCount(0);
    await expect(page.locator('.explanation-square')).toHaveCount(2);
    await expect(page.getByRole('heading', { name: 'Verified tactical evidence' })).toBeVisible();
    await expect(page.locator('.strategic-context a')).toContainText('§3.2');
    expect(await fits(page)).toBe(true);
    await page.locator('#explanation-question').fill('');
    await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); });
    await capture('analysis-populated');

    for (const title of ['Detailed analysis', 'Methodology and limitations']) {
      const summary = page.locator('summary').filter({ hasText: title });
      await summary.focus();
      await page.keyboard.press(`Shift+${tabKey}`);
      await page.keyboard.press(tabKey);
      await expect(summary).toBeFocused();
      await page.keyboard.press('Space');
      await expect(summary.locator('..')).toHaveAttribute('open', '');
      expect(await summary.evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
    }
    await expect(page.getByText('Reference only', { exact: true })).toBeVisible();
    expect(await fits(page)).toBe(true);
    if (width === 320) await capture('analysis-expanded');
    await page.locator('.strategic-context a').focus();
    await page.keyboard.press(`Shift+${tabKey}`);
    await page.keyboard.press(tabKey);
    await expect(page.locator('.strategic-context a')).toBeFocused();
    expect(await page.locator('.strategic-context a').evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
    // The new native radio group is keyboard-operable in all three engines.
    await page.getByRole('radio', { name: 'You go first' }).focus();
    await page.keyboard.press('ArrowRight');
    await expect(page.getByRole('radio', { name: 'AI goes first' })).toBeChecked();
    await expect(page.locator('.board-legend')).toContainText('You · red');
    await page.getByRole('button', { name: 'Start new game' }).click();
    await expect(page.locator('#loading')).toBeHidden();
    await expect(page.locator('.circle.x')).toHaveCount(1);
    await expect(page.locator('.circle.o')).toHaveCount(0);
    await expect(page.locator('.board-legend')).toContainText('You · yellow');
    await expect(page.locator('.settings-note')).toContainText('AI moves first');
    await expect(page.locator('#explain-last')).toHaveAccessibleName('Analyze Last AI Move');
    expect(errors).toEqual([]);
  });
}
