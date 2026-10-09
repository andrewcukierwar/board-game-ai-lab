import { test, expect } from '@playwright/test';

// Run this file separately for each of the four local flag combinations.
const uiEnabled = process.env.VICTOR_RELEASE_UI === 'true';
const apiEnabled = process.env.VICTOR_RELEASE_API === 'true';
const api = process.env.PLAYWRIGHT_API_URL || '';

test('Victor flag combination preserves normal selection and clear errors', async ({ page }) => {
  await page.goto('/connect4');
  const option = page.locator('input[value="victor_research"]');
  await expect(option).toHaveCount(uiEnabled ? 1 : 0);
  if (uiEnabled) {
    await option.check();
    await expect(page.locator('.agent-description')).toContainText('not perfect play');
    if (!apiEnabled) {
      await page.getByRole('button', { name: 'Start game', exact: true }).click();
      await expect(page.getByRole('status')).toContainText('not enabled on this game server');
      await page.locator('input[value="random"]').check();
    } else {
      await page.locator('input[value="random"]').check();
    }
  }
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('Your turn');
  await page.locator('.cell[data-column="3"]:enabled').first().click();
  await expect(page.locator('#loading')).toBeHidden();
  await expect(page.locator('.circle.x')).toHaveCount(1);
  await expect(page.locator('.circle.o')).toHaveCount(1);
});

test.describe('enabled experimental gameplay', () => {
  test.skip(!uiEnabled || !apiEnabled, 'Requires both local flags');
  test('busy AI request leaves its revision unchanged and supports explicit retry', async ({ page }) => {
    await page.goto('/connect4');
    await page.locator('input[value="victor_research"]').check();
    const response = page.waitForResponse(r => r.url().endsWith('/start_game'));
    await page.getByRole('button', { name: 'Start game', exact: true }).click();
    const initial = await (await response).json();
    let rejected = false;
    await page.route('**/v1/connect4/make_move', route => {
      if (!rejected && !('column' in route.request().postDataJSON())) {
        rejected = true;
        return route.fulfill({ status: 503, json: { code: 'agent_busy', error: 'Busy' } });
      }
      return route.continue();
    });
    await page.locator('.cell[data-column="3"]:enabled').first().click();
    await expect(page.getByRole('button', { name: 'Retry AI move' })).toBeVisible();
    await expect(page.getByRole('status')).toContainText('busy with another game');
    const before = await (await page.request.get(`${api}/v1/connect4/games/${initial.game_id}`)).json();
    expect(before.revision).toBe(1);
    await page.getByRole('button', { name: 'Retry AI move' }).click();
    await expect(page.getByRole('status')).toContainText('Your turn');
    const after = await (await page.request.get(`${api}/v1/connect4/games/${initial.game_id}`)).json();
    expect(after.revision).toBe(2);
  });
  for (const first of ['human', 'ai']) {
    test(`complete game with ${first} moving first and reconcile a lost AI response`, async ({ page }) => {
      await page.goto('/connect4');
      await page.locator('input[value="victor_research"]').check();
      await page.locator(`#first-${first}`).check();
      let lost = false;
      let aiRequests = 0;
      await page.route('**/v1/connect4/make_move', async route => {
        if (!('column' in route.request().postDataJSON())) aiRequests++;
        if (!lost && !('column' in route.request().postDataJSON())) {
          lost = true;
          await route.fetch();
          await route.abort('failed');
        } else await route.continue();
      });
      const response = page.waitForResponse(r => r.url().endsWith('/start_game'));
      await page.getByRole('button', { name: 'Start game', exact: true }).click();
      let state = await (await response).json();
      for (let turn = 0; turn < 42; turn++) {
        await expect(page.locator('#loading')).toBeHidden();
        state = await (await page.request.get(`${api}/v1/connect4/games/${state.game_id}`)).json();
        if (state.gameOver) break;
        // Recovery retains the error notice while the reconciled human board
        // becomes playable; a successful next move clears that notice.
        await expect(page.locator(`.cell[data-column="${state.legalMoves[0]}"]:enabled`).first()).toBeEnabled();
        await page.locator(`.cell[data-column="${state.legalMoves[0]}"]:enabled`).first().click();
      }
      expect(state.gameOver).toBe(true);
      expect(lost).toBe(true);
      const history = await (await page.request.get(`${api}/v1/connect4/games/${state.game_id}/history`)).json();
      expect(history.moves).toHaveLength(state.revision);
      expect(aiRequests).toBe(history.moves.filter(record => record.agent.type === 'victor_research').length);
      for (const [index, record] of history.moves.entries()) expect(record.revision).toBe(index + 1);
      await expect(page.getByRole('button', { name: 'Retry AI move' })).toHaveCount(0);
    });
  }
});
