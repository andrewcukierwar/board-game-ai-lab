// Contract: lexicographic UTF-16 key order; ECMAScript JSON primitive spelling.
// This is a project canonical JSON format, not a claim of RFC 8785 compliance.
export function canonicalJSON(value) {
  const ancestors = new Set();
  function visit(item) {
    if (item === null || typeof item === 'string' || typeof item === 'boolean') return JSON.stringify(item);
    if (typeof item === 'number') {
      if (!Number.isFinite(item)) throw new Error('Canonical JSON requires finite numbers.');
      return JSON.stringify(item); // -0 becomes 0, per JSON; no lossy rounding.
    }
    if (typeof item !== 'object') throw new Error('Canonical JSON rejects undefined and non-JSON values.');
    if (ancestors.has(item)) throw new Error('Canonical JSON rejects cycles.');
    if (Object.getOwnPropertySymbols(item).length) throw new Error('Canonical JSON rejects symbol keys.');
    ancestors.add(item);
    let result;
    if (Array.isArray(item)) {
      if (Object.getPrototypeOf(item) !== Array.prototype) throw new Error('Canonical JSON requires ordinary arrays.');
      if (Object.keys(item).length !== item.length) throw new Error('Canonical JSON rejects sparse arrays or extra array properties.');
      if (Reflect.ownKeys(item).some(key => key !== 'length' && (!/^\d+$/.test(key) || Number(key) >= item.length || !('value' in Object.getOwnPropertyDescriptor(item, key))))) throw new Error('Canonical JSON rejects array accessors or extra properties.');
      result = `[${Array.from(item, visit).join(',')}]`;
    } else {
      if (![Object.prototype, null].includes(Object.getPrototypeOf(item))) throw new Error('Canonical JSON requires plain objects.');
      const keys = Reflect.ownKeys(item);
      if (keys.some(key => !Object.getOwnPropertyDescriptor(item, key).enumerable || !('value' in Object.getOwnPropertyDescriptor(item, key)))) {
        throw new Error('Canonical JSON rejects accessors and non-enumerable properties.');
      }
      result = `{${keys.sort().map(key => `${JSON.stringify(key)}:${visit(item[key])}`).join(',')}}`;
    }
    ancestors.delete(item);
    return result;
  }
  return visit(value);
}

export async function browserSHA256(serialized, crypto = globalThis.crypto) {
  if (!crypto?.subtle?.digest) throw new Error('Web Crypto is unavailable. Use HTTPS or localhost, then retry the export.');
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(serialized));
  return Array.from(new Uint8Array(bytes), byte => byte.toString(16).padStart(2, '0')).join('');
}

export async function evidenceDigest(payload, digest = browserSHA256) {
  const hex = await digest(canonicalJSON(payload));
  if (typeof hex !== 'string' || !/^[0-9a-f]{64}$/.test(hex)) throw new Error('SHA-256 implementation did not return lowercase hexadecimal.');
  return hex;
}
