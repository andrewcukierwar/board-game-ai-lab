import { test, expect } from '@playwright/test';

async function start(page) {
  await page.goto('/connect4');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.locator('#message')).toContainText('Your turn');
}

function explained(request) {
  return { ...request, column: request.column ?? null, cached: false, explanation: {
    facts: [{ id: 'position', text: 'This position was verified.', classification: 'confirmed_tactical' }],
    strategic_context: [{ title: 'Immediate tactics', classification: 'context_only', text: 'Check immediate replies.',
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
  await expect(page.locator('#explanation-result')).toContainText('Verified tactical facts');
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
