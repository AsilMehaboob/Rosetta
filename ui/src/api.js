// The Rosetta API address comes from VITE_API_URL (see .env.example).
export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function call(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...options,
      headers: options.body ? { "Content-Type": "application/json" } : undefined,
    });
  } catch {
    throw new Error(`Could not reach the Rosetta API at ${API_URL}. Is it running?`);
  }
  if (!res.ok) {
    let detail = String(res.status);
    try {
      const data = await res.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      // keep the status code
    }
    throw new Error(`The API returned an error: ${detail}`);
  }
  return res.json();
}

// history: earlier turns, oldest first ({question, sql}), so follow-ups like "now split that by city" make sense
export function askQuestion(question, referenceDate, history = []) {
  const body = { question };
  if (referenceDate) body.reference_date = referenceDate;
  if (history.length) body.history = history;
  return call("/questions", { method: "POST", body: JSON.stringify(body) });
}

// evaluation harness
export const getBenchmark = () => call("/eval/benchmark");
export const listEvaluations = () => call("/eval/results");
export const getEvaluation = (runId) => call(`/eval/results/${encodeURIComponent(runId)}`);
export const getEvalStatus = () => call("/eval/status");

export function startEvaluation(runs, only, referenceDate) {
  const body = { runs };
  if (only?.length) body.only = only;
  if (referenceDate) body.reference_date = referenceDate;
  return call("/eval/run", { method: "POST", body: JSON.stringify(body) });
}
