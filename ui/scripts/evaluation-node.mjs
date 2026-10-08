import { createHash } from 'node:crypto';
export const nodeSHA256 = serialized => createHash('sha256').update(serialized, 'utf8').digest('hex');
