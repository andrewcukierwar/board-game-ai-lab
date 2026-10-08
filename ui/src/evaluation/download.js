export function downloadText(text, filename, type, { document = globalThis.document, URL = globalThis.URL, Blob = globalThis.Blob, schedule = setTimeout } = {}) {
  const url = URL.createObjectURL(new Blob([text], { type })), anchor = document.createElement('a');
  try { anchor.href = url; anchor.download = filename; document.body.appendChild(anchor); anchor.click(); }
  finally { anchor.remove(); schedule(() => URL.revokeObjectURL(url), 1000); }
}
