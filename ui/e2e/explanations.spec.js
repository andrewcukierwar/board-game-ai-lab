import { test, expect } from '@playwright/test';

async function start(page) {
  await page.goto('/connect4');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.locator('#message')).toContainText('Your turn');
}

function explained(request) {
  return { ...request, column: request.column ?? null, cached: false, explanation: {
    facts: [{ id: 'position', text: 'This position was verified.', classification: 'confirmed_tactical' }],
    summary: { text: 'Concise verified answer.', focus_id: 'quiet', fact_ids: ['position'] },
    key_facts: [{ id: 'position', text: 'This position was verified.', classification: 'confirmed_tactical' }],
    relevant_squares: [], additional_context: [],
    strategic_context: [{ connection: 'This connects to the verified position.', title: 'Immediate tactics', classification: 'context_only', text: 'Check immediate replies.',
      limitations: ['No long-term proof.'], source: { references: [{ chapter: 3, section: '3.4', thesis_pages: [21, 24] }] } }],
    limitations: ['Post-hoc analysis; agent intent is unknown.'],
  } };
}

test('all explanation modes, sources and explicit failure retry with gameplay available', async ({ page }) => {
  let requests = [];
  await page.route('**/v1/connect4/explain', async route => {
    const body = route.request().postDataJSON();
    requests.push(body);
    await route.fulfill({ json: explained(body) });
  });
  await start(page);
  expect(requests).toHaveLength(0);
  await page.getByRole('button', { name: 'Explain Last Move' }).click();
  await expect(page.locator('#explanation-result')).toContainText('Key tactical evidence');
  await expect(page.locator('#explanation-result a')).toContainText('§3.4, thesis/PDF pp. 21–24');
  await page.getByRole('button', { name: 'Analyze Position', exact: true }).click();
  await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
  await page.getByLabel('Hypothetical move:').selectOption('3');
  await page.getByRole('button', { name: 'What If?', exact: true }).click();
  await expect(page.locator('#explanation-status')).toContainText('Hypothetical Column 4');
  expect(requests.map(r => r.mode)).toEqual(['last_move', 'position', 'what_if']);
  expect(requests[2].column).toBe(3);
  await page.unroute('**/v1/connect4/explain');
  await page.route('**/v1/connect4/explain', route => route.fulfill({ status: 503, json: { error: 'Explanations are unavailable.' } }), { times: 1 });
  await page.getByRole('button', { name: 'Analyze Position', exact: true }).click();
  await expect(page.locator('#explanation-status')).toContainText('unavailable');
  await expect(page.locator('.cell:enabled').first()).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Start new game' })).toBeEnabled();
  await page.route('**/v1/connect4/explain', route => route.fulfill({ json: explained(route.request().postDataJSON()) }));
  await page.getByRole('button', { name: 'Analyze Position', exact: true }).click();
  await expect(page.locator('#explanation-result')).toContainText('This position was verified');
});

for (const action of ['move', 'restart', 'navigate']) {
  test(`${action} during explanation loading cancels and prevents stale rendering`, async ({ page }) => {
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    let began;
    const entered = new Promise(resolve => { began = resolve; });
    await page.route('**/v1/connect4/explain', async route => {
      began();
      await gate;
      try { await route.fulfill({ json: explained(route.request().postDataJSON()) }); } catch { /* canceled */ }
    });
    await start(page);
    await page.getByRole('button', { name: 'Analyze Position', exact: true }).click();
    await entered;
    await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'true');
    await expect(page.getByRole('button', { name: 'Analyze Position', exact: true })).toBeDisabled();
    if (action === 'move') {
      await page.locator('.cell[data-column="3"]:enabled').first().click();
      await expect(page.locator('#loading')).toBeHidden();
    } else if (action === 'restart') {
      await page.getByRole('button', { name: 'Start new game' }).click();
      await expect(page.locator('#message')).toContainText('Your turn');
    } else {
      await page.getByRole('link', { name: 'Home', exact: true }).click();
      await expect(page.getByRole('link', { name: 'Play Connect 4' })).toBeVisible();
    }
    release();
    if (action !== 'navigate') {
      await expect(page.locator('#explanation-result')).toBeEmpty();
      await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
      await expect(page.getByRole('button', { name: 'Analyze Position', exact: true })).toBeEnabled();
    }
  });
}

test('disabled backend returns graceful explanation error while actual gameplay works', async ({ page }) => {
  // Intercept even this failure path so no test can accidentally call a paid provider.
  await page.route('**/v1/connect4/explain', route => route.fulfill({
    status: 503, json: { error: 'Explanations are disabled. You can continue playing.', code: 'explanations_disabled' },
  }));
  await start(page);
  await page.getByRole('button', { name: 'Analyze Position', exact: true }).click();
  await expect(page.locator('#explanation-status')).toContainText(/disabled|unavailable/);
  await page.locator('.cell[data-column="3"]:enabled').first().click();
  await expect(page.locator('#loading')).toBeHidden();
  await expect(page.locator('.circle.x')).toHaveCount(1);
  await expect(page.locator('.circle.o')).toHaveCount(1);
  await expect(page.getByRole('button', { name: 'Explain AI Move' })).toBeEnabled();
});

test('a response for another hypothetical column is rejected without affecting play', async ({ page }) => {
  await page.route('**/v1/connect4/explain', route => route.fulfill({
    json: { ...explained(route.request().postDataJSON()), column: 0 },
  }));
  await start(page);
  await page.getByLabel('Hypothetical move:').selectOption('3');
  await page.getByRole('button', { name: 'What If?', exact: true }).click();
  await expect(page.locator('#explanation-status')).toContainText('could not be loaded');
  await expect(page.locator('#explanation-result')).toBeEmpty();
  await expect(page.locator('.cell:enabled').first()).toBeEnabled();
});

test('revision 36 summary, expandable evidence and square highlighting preserve gameplay', async ({ page }) => {
  const api = process.env.PLAYWRIGHT_API_URL || '';
  let state = await (await page.request.post(`${api}/v1/connect4/start_game`, { data: {
    player1: { type: 'human' }, player2: { type: 'human' },
  } })).json();
  for (const column of [2, 5, 6, 3, 5, 5, 0, 2, 3, 2, 6, 0, 4, 4, 3, 6, 2, 2,
    4, 3, 6, 0, 0, 2, 6, 5, 4, 3, 3, 0, 5, 4, 0, 6, 5, 4]) {
    const moved = await page.request.post(`${api}/v1/connect4/make_move`, { data: {
      game_id: state.game_id, revision: state.revision, column,
    } });
    expect(moved.status()).toBe(200);
    state = await moved.json();
  }
  expect(state.legalMoves).toEqual([1]);
  await page.route('**/v1/connect4/start_game', route => route.fulfill({ status: 201, json: state }), { times: 1 });
  await page.route('**/v1/connect4/explain', route => {
    const request = route.request().postDataJSON();
    const result = explained(request);
    const data = result.explanation;
    data.summary = { focus_id: 'forced_reply_b2', fact_ids: ['gravity'],
      text: "Column 2 is Player 1's only available move, but playing there places the piece on b1. This makes b2 accessible to Player 2, who can immediately play there to complete four in a row." };
    data.facts = [{ id: 'gravity', classification: 'confirmed_tactical', text: 'Playing b1 makes the winning square b2 gravity-playable.' }];
    data.key_facts = [...data.facts];
    data.strategic_context = [{ title: 'Threats and winning squares', classification: 'context_only',
      text: 'A winning square must be reachable under gravity.', connection: 'b2 is inaccessible until b1 is filled.',
      limitations: ['Geometric patterns alone do not prove a long-term result.'],
      source: { references: [{ chapter: 3, section: '3.1', thesis_pages: [16, 18] }] } }];
    data.relevant_squares = [
      { name: 'b1', row_index: 5, row: 1, column: 1 },
      { name: 'b2', row_index: 4, row: 2, column: 1 },
    ];
    return route.fulfill({ json: result });
  });
  await start(page);
  await page.getByRole('button', { name: 'Analyze Position', exact: true }).click();
  await expect(page.locator('.primary-explanation')).toContainText('b2 accessible to Player 2');
  await expect(page.locator('#explanation-result > section')).toHaveCount(3);
  await expect(page.locator('.explanation-square')).toHaveCount(2);
  await expect(page.locator('.square-label')).toHaveText(['b2', 'b1']);
  await expect(page.locator('.cell[data-square="b1"]')).toBeEnabled();
  const detailed = page.locator('details').filter({ has: page.getByText('Detailed analysis', { exact: true }) });
  const methodology = page.locator('details').filter({ has: page.getByText('Methodology and limitations', { exact: true }) });
  await expect(detailed).not.toHaveAttribute('open');
  await expect(methodology).not.toHaveAttribute('open');
  await detailed.locator('summary').click();
  await expect(detailed.getByText('Complete verified tactical facts')).toBeVisible();
  await detailed.locator('summary').click();
  await expect(detailed.getByText('Complete verified tactical facts')).toBeHidden();
  await methodology.locator('summary').click();
  await expect(methodology.getByText('Post-hoc analysis; agent intent is unknown.')).toBeVisible();
  await methodology.locator('summary').click();
  await page.screenshot({ path: '/private/tmp/phase3b1-panel.png', fullPage: true });
  const actual = await (await page.request.get(`${api}/v1/connect4/games/${state.game_id}`)).json();
  expect(actual).toEqual(state);
  await page.locator('.cell[data-square="b1"]').click();
  await expect(page.locator('#loading')).toBeHidden();
  await expect(page.locator('#explanation-result')).toBeEmpty();
  await expect(page.locator('.explanation-square')).toHaveCount(0);
  await expect(page.locator('.square-label')).toHaveCount(0);
});
