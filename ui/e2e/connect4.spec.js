import { test, expect } from '@playwright/test';

const api = path => (process.env.PLAYWRIGHT_API_URL || '') + path;

async function start(page, opponent = 'negamax') {
  await page.goto('/');
  await page.getByRole('link', { name: 'Play Connect 4' }).click();
  await page.getByLabel('Opponent:').selectOption(opponent);
  const result = page.waitForResponse(r => r.url().endsWith('/start_game') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  const response = await result;
  expect(response.status()).toBe(201);
  await expect(page.getByRole('status')).toContainText('Your turn');
  return response.json();
}

for (const opponent of ['random', 'negamax']) {
  test(`complete a browser game against ${opponent}, then restart with a different opponent`, async ({ page }) => {
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    let state = await start(page, opponent);
    for (let turns = 0; turns < 42 && !state.gameOver; turns++) {
      const column = state.legalMoves[0];
      const humanResult = page.waitForResponse(r => r.url().endsWith('/make_move') && r.request().method() === 'POST' && 'column' in r.request().postDataJSON());
      await page.locator(`.cell[data-column="${column}"]:enabled`).first().click();
      const human = await humanResult;
      expect(human.status()).toBe(200);
      // Wait until the entire human + AI request chain has settled.
      await expect(page.locator('#loading')).toBeHidden();
      const result = await page.request.get(api(`/v1/connect4/games/${state.game_id}`));
      state = await result.json();
      if (!state.gameOver) await expect(page.getByRole('status')).toContainText('Your turn');
    }
    expect(state.gameOver).toBe(true);
    await expect(page.locator('.cell:enabled')).toHaveCount(0);
    await expect(page.getByRole('status')).toContainText(/wins|win|draw/);
    const previousId = state.game_id;
    const nextOpponent = opponent === 'random' ? 'negamax' : 'random';
    await page.getByLabel('Opponent:').selectOption(nextOpponent);
    const restarted = page.waitForResponse(r => r.url().endsWith('/start_game') && r.request().method() === 'POST');
    await page.getByRole('button', { name: 'Start new game' }).click();
    state = await (await restarted).json();
    expect(state.game_id).not.toBe(previousId);
    expect(state.players[1].type).toBe(nextOpponent);
    expect(state.revision).toBe(0);
    expect((await page.request.get(api(`/v1/connect4/games/${previousId}`))).status()).toBe(404);
    await expect(page.getByRole('status')).toContainText('Your turn');
    // Exercise unmount/remount and a direct SPA refresh.
    await page.getByRole('link', { name: 'Home', exact: true }).click();
    await page.getByRole('link', { name: 'Play Connect 4' }).click();
    await page.reload();
    await expect(page.getByRole('button', { name: 'Start game', exact: true })).toBeVisible();
    expect(pageErrors).toEqual([]);
  });
}

test('two browser sessions stay independent', async ({ page, browser }) => {
  const secondContext = await browser.newContext();
  const second = await secondContext.newPage();
  try {
    const firstState = await start(page, 'random');
    const secondState = await start(second, 'negamax');
    expect(firstState.game_id).not.toBe(secondState.game_id);
    await page.locator('.cell[data-column="3"]:enabled').first().click();
    await expect(page.locator('#loading')).toBeHidden();
    expect((await page.request.get(api(`/v1/connect4/games/${firstState.game_id}`))).status()).toBe(200);
    const secondActual = await (await second.request.get(api(`/v1/connect4/games/${secondState.game_id}`))).json();
    expect(secondActual).toEqual(secondState);
    await expect(second.locator('.circle.x')).toHaveCount(0);
    await expect(second.locator('.circle.o')).toHaveCount(0);
  } finally {
    await secondContext.close();
  }
});

test('failed initialization and AI request recover without reload or automatic retry loop', async ({ page }) => {
  await page.goto('/connect4');
  await page.route('**/v1/connect4/start_game', route => route.fulfill({
    status: 503, contentType: 'application/json', body: JSON.stringify({ error: 'Temporarily unavailable' }),
  }), { times: 1 });
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('Temporarily unavailable');
  await expect(page.getByRole('button', { name: 'Start game', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('Your turn');
  let failures = 0;
  await page.route('**/v1/connect4/make_move', route => {
    if (!('column' in route.request().postDataJSON()) && failures++ === 0) {
      return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ error: 'AI temporarily unavailable' }) });
    }
    return route.continue();
  });
  await page.locator('.cell[data-column="3"]:enabled').first().click();
  await expect(page.getByRole('button', { name: 'Retry AI move' })).toBeVisible();
  expect(failures).toBe(1);
  await page.getByRole('button', { name: 'Retry AI move' }).click();
  await expect(page.getByRole('status')).toContainText('Your turn');
  await expect(page.locator('.circle.x')).toHaveCount(1);
  await expect(page.locator('.circle.o')).toHaveCount(1);
});

test('a removed or expired game can be replaced from the browser', async ({ page }) => {
  const state = await start(page);
  // Replacing it server-side reproduces the same 404 contract as TTL expiration.
  const replacement = await page.request.post(api('/v1/connect4/start_game'), { data: { replace_game_id: state.game_id } });
  expect(replacement.status()).toBe(201);
  await page.locator('.cell[data-column="3"]:enabled').first().click();
  await expect(page.getByRole('status')).toContainText('expired');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('Your turn');
});

test('a slow server startup keeps Start locked until the response arrives', async ({ page }) => {
  await page.goto('/connect4');
  let release;
  const ready = new Promise(resolve => { release = resolve; });
  await page.route('**/v1/connect4/start_game', async route => {
    await ready;
    await route.continue();
  });
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.locator('#loading')).toContainText('wakes up');
  await expect(page.getByRole('button', { name: 'Start game', exact: true })).toBeDisabled();
  release();
  await expect(page.getByRole('status')).toContainText('Your turn');
});

test('a cold-start HTML response allows an explicit fresh start', async ({ page }) => {
  await page.goto('/connect4');
  await page.route('**/v1/connect4/start_game', route => route.fulfill({
    status: 200, contentType: 'text/html', body: '<html>Waking up</html>',
  }), { times: 1 });
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('waking up');
  await expect(page.getByRole('button', { name: 'Start game', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('Your turn');
});

test('a lost AI response after commit recovers the board without a duplicate move', async ({ page }) => {
  await start(page, 'random');
  let aiRequests = 0;
  await page.route('**/v1/connect4/make_move', async route => {
    if (!('column' in route.request().postDataJSON())) {
      aiRequests++;
      await route.fetch(); // The server commits, then the browser loses the response.
      await route.abort('failed');
    } else {
      await route.continue();
    }
  });
  await page.locator('.cell[data-column="3"]:enabled').first().click();
  await expect(page.locator('#loading')).toBeHidden();
  await expect(page.locator('.circle.x')).toHaveCount(1);
  await expect(page.locator('.circle.o')).toHaveCount(1);
  await expect(page.locator('#retry-button')).toBeHidden();
  await expect(page.locator('.cell:enabled')).toHaveCount(42);
  expect(aiRequests).toBe(1);
});
