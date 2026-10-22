// The Rosetta API address comes from VITE_API_URL (see .env.example).
export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

export async function askQuestion(question, referenceDate) {
  const body = { question };
  if (referenceDate) body.reference_date = referenceDate;

  let res;
  try {
    res = await fetch(`${API_URL}/questions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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
