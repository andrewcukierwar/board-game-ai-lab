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
  await page.getByRole('button', { name: 'Analyze Last Move' }).click();
  await expect(page.locator('#explanation-result')).toContainText('Verified tactical evidence');
  await expect(page.locator('#explanation-result a')).toContainText('§3.4, thesis/PDF pp. 21–24');
  await page.getByRole('button', { name: 'Analyze Position', exact: true }).click();
  await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
  await page.getByRole('button', { name: 'What If?', exact: true }).click();
  await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
  const defaultColumn = Number(await page.getByLabel('Hypothetical move:').inputValue());
  await page.getByLabel('Hypothetical move:').selectOption('3');
  await page.getByRole('button', { name: 'What If?', exact: true }).click();
  await expect(page.locator('#explanation-status')).toContainText('Hypothetical Column 4');
  expect(requests.map(r => r.mode)).toEqual(['last_move', 'position', 'what_if', 'what_if']);
  expect(requests[2].column).toBe(defaultColumn);
  expect(requests[3].column).toBe(3);
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
  await expect(page.getByRole('button', { name: 'Analyze Last AI Move' })).toBeEnabled();
});

test('a response for another hypothetical column is rejected without affecting play', async ({ page }) => {
  await page.route('**/v1/connect4/explain', route => route.fulfill({
    json: { ...explained(route.request().postDataJSON()), column: 0 },
  }));
  await start(page);
  await page.getByRole('button', { name: 'What If?', exact: true }).click();
  await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
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
  // Settings rerenders must preserve React-owned highlights.
  await page.locator('input[name="opponent"][value="random"]').check();
  await expect(page.locator('.explanation-square')).toHaveCount(2);
  await expect(page.locator('.square-label')).toHaveText(['b2', 'b1']);
  await expect(page.locator('.cell:enabled')).toHaveCount(6);
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

  const actual = await (await page.request.get(`${api}/v1/connect4/games/${state.game_id}`)).json();
  expect(actual).toEqual(state);
  await page.locator('.cell[data-square="b1"]').click();
  await expect(page.locator('#loading')).toBeHidden();
  await expect(page.locator('#explanation-result')).toBeEmpty();
  await expect(page.locator('.explanation-square')).toHaveCount(0);
  await expect(page.locator('.square-label')).toHaveCount(0);
});

function representative(request) {
  const result = explained(request);
  const data = result.explanation;
  data.summary.text = 'This move makes d2 reachable under gravity. The resulting position allows an immediate reply there; avoiding that reply does not prove a long-term win.';
  data.facts = [
    { id: 'position', text: 'The highlighted square d2 becomes reachable when d1 is occupied.', classification: 'confirmed_tactical' },
    { id: 'reply', text: 'Immediate replies were checked against legal columns. Geometric completion squares are not automatically playable.', classification: 'confirmed_tactical' },
  ];
  data.key_facts = data.facts;
  data.relevant_squares = [{ name: 'd1', column: 3, row: 1, row_index: 5 }, { name: 'd2', column: 3, row: 2, row_index: 4 }];
  data.strategic_context[0] = { ...data.strategic_context[0], title: 'Threats and winning squares',
    text: 'A winning square must be reachable under gravity.', connection: 'One relevant strategic concept is the distinction between a completion square and a playable winning square.',
    preconditions: ['The reference applies to a legal position with gravity.'] };
  data.additional_context = [{ ...data.strategic_context[0], title: 'Formal coverage', classification: 'reference_only',
    text: 'Formal coverage requires compatible solutions and additional preconditions.', connection: '',
    limitations: ['No formal rule application is established by this analysis.'] }];
  return result;
}

for (const [name, width, height] of [['desktop', 1440, 1100], ['tablet', 820, 1100], ['mobile', 375, 812], ['narrow-mobile', 320, 740]]) {
  test(`populated analysis and disclosures fit at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height });
    await page.route('**/v1/connect4/explain', route => route.fulfill({ json: representative(route.request().postDataJSON()) }));
    await start(page);
    await page.locator('#explain-last').focus();
    await page.keyboard.press('Tab');
    await expect(page.locator('#analyze-position')).toBeFocused();
    expect(await page.locator('#analyze-position').evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
    await page.keyboard.press('Enter');
    await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
    await expect(page.locator('#analyze-position')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('.explanation-square')).toHaveCount(2);
    expect(await page.locator('.square-label').evaluateAll(nodes => nodes.every(node => node.getAttribute('aria-hidden') === 'true'))).toBe(true);
    const source = page.locator('.strategic-context a');
    await expect(source).toHaveAttribute('href', 'https://tromp.github.io/c4/connect4_thesis.pdf');
    await expect(source).toHaveAttribute('target', '_blank');
    await expect(source).toHaveAttribute('rel', 'noopener noreferrer');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: testInfo.outputPath(`analysis-${name}-result.png`), fullPage: true });
    await page.locator('#explanation-panel').screenshot({ path: testInfo.outputPath(`analysis-${name}-panel.png`) });
    for (const title of ['Detailed analysis', 'Methodology and limitations']) {
      const summary = page.getByText(title, { exact: true });
      await summary.focus();
      expect(await summary.evaluate(node => getComputedStyle(node).outlineStyle)).toBe('solid');
      await page.keyboard.press('Space');
      await expect(summary.locator('..')).toHaveAttribute('open', '');
    }
    await expect(page.getByText('Reference only', { exact: true })).toBeVisible();
    await expect(page.getByText('Reference precondition: The reference applies to a legal position with gravity.', { exact: true })).toHaveCount(2);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(await page.locator('.skip-link').evaluate(node => node.getBoundingClientRect().bottom)).toBeLessThanOrEqual(0);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.locator('#explanation-panel').screenshot({ path: testInfo.outputPath(`analysis-${name}-expanded.png`) });
    await page.locator('#what-if').click();
    await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
    await page.locator('#what-if-column').selectOption('3');
    await page.locator('#what-if').focus(); await page.keyboard.press('Space');
    await expect(page.locator('#what-if')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#explanation-status')).toContainText('Hypothetical Column 4');
  });
}

test('optional question is capped at 500 and submitted unchanged', async ({ page }) => {
  let body;
  await page.route('**/v1/connect4/explain', route => {
    body = route.request().postDataJSON();
    return route.fulfill({ json: explained(body) });
  });
  await start(page);
  await page.getByLabel('Optional question', { exact: true }).fill('q'.repeat(500));
  await page.getByLabel('Optional question', { exact: true }).press('End');
  await page.getByLabel('Optional question', { exact: true }).press('x');
  await expect(page.locator('#question-count')).toHaveText('500/500');
  await page.locator('#analyze-position').click();
  await expect(page.locator('#explanation-result')).toContainText('Concise verified answer');
  expect(body.question).toBe('q'.repeat(500));
  expect(body).not.toHaveProperty('column');
});

for (const wrong of ['game_id', 'revision', 'mode', 'explanation']) {
  test(`invalid ${wrong} never renders result or highlights`, async ({ page }) => {
    await page.route('**/v1/connect4/explain', route => {
      const result = representative(route.request().postDataJSON());
      result[wrong] = wrong === 'revision' ? result.revision + 1 : wrong === 'explanation' ? { facts: [] } : 'wrong';
      return route.fulfill({ json: result });
    });
    await start(page); await page.locator('#analyze-position').click();
    await expect(page.locator('#explanation-status')).toContainText('Gameplay remains available');
    await expect(page.locator('#explanation-result')).toBeEmpty();
    await expect(page.locator('.explanation-square')).toHaveCount(0);
    await page.locator('.cell[data-column="3"]:enabled').first().click();
    await expect(page.locator('.circle.x')).toHaveCount(1);
    await expect(page.locator('.circle.o')).toHaveCount(1);
    await page.locator('#restart-button').click();
    await expect(page.locator('.circle.x')).toHaveCount(0);
  });
}

for (const action of ['restart', 'navigate']) {
  test(`${action} clears a completed analysis and board labels`, async ({ page }) => {
    await page.route('**/v1/connect4/explain', route => route.fulfill({ json: representative(route.request().postDataJSON()) }));
    await start(page); await page.locator('#analyze-position').click();
    await expect(page.locator('.explanation-square')).toHaveCount(2);
    if (action === 'restart') await page.locator('#restart-button').click();
    else {
      await page.getByRole('link', { name: 'Home', exact: true }).click();
      await page.getByRole('link', { name: 'Play Connect 4', exact: true }).click();
    }
    await expect(page.locator('.explanation-square, .square-label')).toHaveCount(0);
    await expect(page.locator('#explanation-result')).toBeEmpty();
  });
}

test('loading, disabled service and long unavailable error fit at 320px while gameplay stays available', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 320, height: 740 });
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  await page.route('**/v1/connect4/explain', async route => {
    await gate;
    await route.fulfill({ status: 503, json: { error: 'Explanations are disabled. You can continue playing.' } });
  });
  await start(page); await page.locator('#analyze-position').click();
  await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'true');
  await expect(page.locator('#explanation-status')).toContainText('Preparing grounded analysis');
  await expect(page.locator('.cell:enabled').first()).toBeEnabled();
  await expect(page.locator('#restart-button')).toBeEnabled();
  await expect(page.getByLabel('Optional question', { exact: true })).toBeDisabled();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: testInfo.outputPath('analysis-loading-320.png'), fullPage: true });
  release();
  await expect(page.locator('#explanation-status')).toContainText('disabled');
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: testInfo.outputPath('analysis-disabled-320.png'), fullPage: true });
  await page.unroute('**/v1/connect4/explain');
  await page.route('**/v1/connect4/explain', route => route.fulfill({ status: 429, json: {
    error: 'Analysis is temporarily unavailable. Please try again later. '.repeat(8) + 'LongUnbrokenDiagnosticLabel'.repeat(8),
  } }));
  await page.locator('#analyze-position').click();
  await expect(page.locator('#explanation-status')).toContainText('Gameplay remains available');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: testInfo.outputPath('analysis-long-error-320.png'), fullPage: true });
  await page.locator('.cell[data-column="3"]:enabled').first().click();
  await expect(page.locator('.circle.x')).toHaveCount(1);
  await expect(page.locator('.circle.o')).toHaveCount(1);
});

test('what-if selector leaves the form and layout in other modes but retains its legal selection', async ({ page }) => {
  const requests = [];
  await page.route('**/v1/connect4/explain', route => {
    const body = route.request().postDataJSON();
    requests.push(body);
    return route.fulfill({ json: explained(body) });
  });
  await start(page);
  for (const action of ['#analyze-position', '#explain-last']) {
    await page.locator(action).click();
    await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
    await expect(page.getByLabel('Hypothetical move:')).toHaveCount(0);
    await expect(page.locator('#column-help')).toHaveCount(0);
    const fields = await page.locator('.analysis-fields').boundingBox();
    const question = await page.locator('.analysis-question').boundingBox();
    expect(question.width).toBeCloseTo(fields.width, 0);
    expect(requests.at(-1)).not.toHaveProperty('column');
  }
  await page.locator('#what-if').click();
  await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
  await page.getByLabel('Hypothetical move:').selectOption('3');
  for (const action of ['#explain-last', '#analyze-position']) {
    await page.locator(action).click();
    await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
    await expect(page.getByLabel('Hypothetical move:')).toHaveCount(0);
    await page.locator('#what-if').click();
    await expect(page.locator('#explanation-panel')).toHaveAttribute('aria-busy', 'false');
    await expect(page.getByLabel('Hypothetical move:')).toHaveValue('3');
    await expect(page.locator('#explanation-status')).toContainText('Hypothetical Column 4');
    expect(requests.at(-1).column).toBe(3);
  }
});
