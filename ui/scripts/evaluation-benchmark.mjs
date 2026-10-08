import { performance } from 'node:perf_hooks';
import { evaluationSeason, DRAW } from '../tests/fixtures/evaluation.js';
import { createEvaluationExport, canonicalPayload } from '../src/evaluation/export.js';
import { canonicalJSON } from '../src/evaluation/canonical.js';
import { gamesCSV, summaryCSV } from '../src/evaluation/csv.js';
import { verifyEvaluationExport } from '../src/evaluation/verify.js';
import { nodeSHA256 } from './evaluation-node.mjs';
const season = evaluationSeason({ size: 12, games: 8, sequences: [DRAW] });
async function timing(fn, repeats = 5) {
  let result; const start = performance.now(); for (let i = 0; i < repeats; i++) result = await fn();
  return { ms: (performance.now() - start) / repeats, result };
}
const build = await timing(() => createEvaluationExport(season, { digest: nodeSHA256 })), artifact = build.result;
const serialized = await timing(() => canonicalJSON(canonicalPayload(artifact)), 20), hash = await timing(() => nodeSHA256(serialized.result), 20);
const webHash = await timing(() => crypto.subtle.digest('SHA-256', new TextEncoder().encode(serialized.result)), 20);
const json = JSON.stringify(artifact, null, 2) + '\n', games = await timing(() => gamesCSV(artifact)), summary = await timing(() => summaryCSV(artifact));
const verify = await timing(() => verifyEvaluationExport(JSON.parse(json), { digest: nodeSHA256 }));
if (!verify.result.ok) throw new Error(JSON.stringify(verify.result.errors));
console.log(JSON.stringify({ synthetic: true, games: 528, plies: 22176, exportConstructionMs: build.ms, canonicalSerializationMs: serialized.ms,
  nodeSHA256Ms: hash.ms, webCryptoSHA256Ms: webHash.ms, jsonBytes: Buffer.byteLength(json), canonicalPayloadBytes: Buffer.byteLength(serialized.result),
  gamesCSVBytes: Buffer.byteLength(games.result), summaryCSVBytes: Buffer.byteLength(summary.result), gamesCSVConstructionMs: games.ms,
  summaryCSVConstructionMs: summary.ms, verifierMs: verify.ms, repeats: { construction: 5, serializationAndHash: 20, csvAndVerification: 5 } }, null, 2));
