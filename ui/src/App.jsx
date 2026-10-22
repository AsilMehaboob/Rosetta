import { useState } from "react";
import { askQuestion } from "./api.js";
import AnswerCard from "./AnswerCard.jsx";
import AskForm from "./AskForm.jsx";
import Evidence from "./Evidence.jsx";
import ResultTable from "./ResultTable.jsx";

const HISTORY_KEY = "rosetta.history";
const HISTORY_MAX = 8;

// Recent answers live only in this browser; reading or writing may fail (private mode), which is fine.
function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function saveHistory(items) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
  } catch {
    // not remembered across reloads
  }
}

export default function App() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState(loadHistory);

  async function ask(question, asOf) {
    setBusy(true);
    setError("");
    try {
      const answer = await askQuestion(question, asOf);
      setResult(answer);
      const next = [answer, ...history.filter((h) => h.question !== answer.question)].slice(0, HISTORY_MAX);
      setHistory(next);
      saveHistory(next);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="page">
      <header className="brand">
        <h1>Rosetta</h1>
        <p>Ask your data a question.</p>
      </header>

      <AskForm onAsk={ask} busy={busy} />

      {history.length > 0 && (
        <nav className="recent" aria-label="Recent questions">
          {history.map((h) => (
            <button
              key={h.question}
              type="button"
              className={`chip ${result?.question === h.question ? "active" : ""}`}
              onClick={() => setResult(h)}
              title={h.question}
            >
              {h.question}
            </button>
          ))}
        </nav>
      )}

      {error && <p className="error">{error}</p>}

      {busy && (
        <div className="thinking">
          <span className="dot" />
          Reading the question, finding the data and checking the query…
        </div>
      )}

      {!busy && result && (
        <>
          <AnswerCard result={result} />
          <ResultTable columns={result.columns} rows={result.rows || []} />
          <Evidence result={result} />
        </>
      )}
    </main>
  );
}
