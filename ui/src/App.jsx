import { useEffect, useRef, useState } from "react";
import { askQuestion } from "./api.js";
import AnswerCard from "./AnswerCard.jsx";
import AskForm from "./AskForm.jsx";
import Evidence from "./Evidence.jsx";
import ResultTable from "./ResultTable.jsx";

const HISTORY_KEY = "rosetta.conversation";
const HISTORY_MAX = 20;

// The conversation lives only in this browser; storage may be unavailable (private mode), which is fine.
function loadTurns() {
  try {
    return (JSON.parse(localStorage.getItem(HISTORY_KEY)) || []).filter((t) => t.status !== "pending");
  } catch {
    return [];
  }
}

function saveTurns(turns) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(turns.filter((t) => t.status !== "pending").slice(-HISTORY_MAX)));
  } catch {
    // not remembered across reloads
  }
}

function Reply({ turn }) {
  if (turn.status === "pending") {
    return (
      <div className="thinking">
        <span className="dot" />
        Reading the question, finding the data and checking the query…
      </div>
    );
  }
  if (turn.status === "error") return <p className="error">{turn.error}</p>;
  const result = turn.result;
  return (
    <>
      <AnswerCard result={result} />
      <ResultTable columns={result.columns} rows={result.rows || []} />
      <Evidence result={result} />
    </>
  );
}

export default function App() {
  const [turns, setTurns] = useState(loadTurns);
  const end = useRef(null);
  const busy = turns.some((t) => t.status === "pending");

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    saveTurns(turns);
  }, [turns]);

  async function ask(question, asOf) {
    const id = Date.now();
    setTurns((ts) => [...ts, { id, question, status: "pending" }]);
    let update;
    try {
      update = { status: "done", result: await askQuestion(question, asOf) };
    } catch (e) {
      update = { status: "error", error: e.message };
    }
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...update } : t)));
  }

  return (
    <div className="app">
      <header className="topbar">
        <span className="logo">Rosetta</span>
        {turns.length > 0 && (
          <button type="button" className="ghost" onClick={() => setTurns([])} disabled={busy}>
            New chat
          </button>
        )}
      </header>

      <main className="thread">
        {turns.length === 0 ? (
          <div className="empty">
            <h1>What would you like to know?</h1>
            <p>Ask in plain words. Rosetta finds the data, shows the answer and how it worked it out.</p>
          </div>
        ) : (
          turns.map((turn) => (
            <section key={turn.id} className="turn">
              <div className="bubble">{turn.question}</div>
              <Reply turn={turn} />
            </section>
          ))
        )}
        <div ref={end} />
      </main>

      <footer className="dock">
        <AskForm onAsk={ask} busy={busy} />
        <p className="fine">Answers come from your database; check the details under each answer.</p>
      </footer>
    </div>
  );
}
