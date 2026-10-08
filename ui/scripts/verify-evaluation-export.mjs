import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { verifyEvaluationExport } from '../src/evaluation/verify.js';
import { nodeSHA256 } from './evaluation-node.mjs';

export async function verifyFile(filename) {
  const artifact = JSON.parse(await readFile(filename, 'utf8'));
  return verifyEvaluationExport(artifact, { digest: nodeSHA256 });
}
async function main() {
  if (process.argv.length !== 3) throw new Error('Usage: node ui/scripts/verify-evaluation-export.mjs path/to/evaluation.json');
  const result = await verifyFile(process.argv[2]);
  if (!result.ok) throw new Error(result.errors.map(e => `${e.stage}: ${e.message}`).join('\n'));
  const s = result.summary;
  console.log(`Verification PASS\nSchema: ${s.schema}\nEvidence SHA-256: ${s.digest}\nField: ${s.field_size} entrants\n${s.status.toUpperCase()}: ${s.completed_games}/${s.scheduled_games} games\nProvenance: ${JSON.stringify(s.provenance)}`);
  for (const warning of result.warnings) console.log(`Provenance notice: ${warning}`);
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main().catch(e => { console.error(`Verification FAIL: ${e.message}`); process.exitCode = 1; });
