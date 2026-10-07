import { readFile } from 'node:fs/promises';
import { transformWithEsbuild } from 'vite';

// Reuse Vite's JSX transform in the existing Node/JSDOM suite, with no new
// testing framework. CSS is verified by the production build and browser suite.
export async function load(url, context, nextLoad) {
  if (url.endsWith('.css')) return { format: 'module', source: '', shortCircuit: true };
  if (!url.endsWith('.jsx')) return nextLoad(url, context);
  const source = await readFile(new URL(url), 'utf8');
  const result = await transformWithEsbuild(source, url, { loader: 'jsx', jsx: 'automatic' });
  return { format: 'module', source: result.code, shortCircuit: true };
}
