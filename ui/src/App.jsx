import { useState } from "react";
import { askQuestion } from "./api.js";
import AnswerCard from "./AnswerCard.jsx";
import AskForm from "./AskForm.jsx";
import Evidence from "./Evidence.jsx";
import ResultTable from "./ResultTable.jsx";

export default function App() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function ask(question, asOf) {
    setBusy(true);
    setError("");
    try {
      setResult(await askQuestion(question, asOf));
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
      {error && <p className="error">{error}</p>}
      {result && (
        <>
          <AnswerCard result={result} />
          <ResultTable columns={result.columns} rows={result.rows || []} />
          <Evidence result={result} />
        </>
      )}
    </main>
  );
}
