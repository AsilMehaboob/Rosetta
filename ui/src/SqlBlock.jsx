import { useState } from "react";

const KEYWORDS = new Set(
  (
    "select from where and or not in is null as on join left right inner outer full cross group by order having " +
    "distinct count sum avg min max case when then else end between like exists union all with over partition " +
    "lag lead row_number rank asc desc date interval limit fetch first rows only offset trunc coalesce nvl cast"
  ).split(" "),
);

// strings, comments, numbers and words, in the order they appear
const TOKEN = /('(?:[^']|'')*')|(--[^\n]*)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_$]*)/g;

function highlight(sql) {
  const parts = [];
  let last = 0;
  for (const m of sql.matchAll(TOKEN)) {
    if (m.index > last) parts.push(sql.slice(last, m.index));
    const [text, str, comment, num, word] = m;
    const kind = str ? "tok-str" : comment ? "tok-comment" : num ? "tok-num" : KEYWORDS.has(word.toLowerCase()) ? "tok-kw" : null;
    parts.push(kind ? <span key={m.index} className={kind}>{text}</span> : text);
    last = m.index + text.length;
  }
  parts.push(sql.slice(last));
  return parts;
}

// A query with simple syntax colouring and a copy button.
export default function SqlBlock({ sql }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(sql);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard not available; the text can still be selected by hand
    }
  }

  return (
    <div className="sql-block">
      <button type="button" className="copy" onClick={copy}>
        {copied ? "Copied" : "Copy"}
      </button>
      <pre className="sql">{highlight(sql)}</pre>
    </div>
  );
}
