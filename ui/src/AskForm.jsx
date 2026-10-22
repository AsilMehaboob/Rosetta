import { useState } from "react";

export default function AskForm({ onAsk, busy }) {
  const [question, setQuestion] = useState("");
  const [asOf, setAsOf] = useState("");

  function submit(e) {
    e.preventDefault();
    if (question.trim() && !busy) onAsk(question.trim(), asOf);
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) submit(e); // Shift+Enter for a new line
  }

  return (
    <form className="ask" onSubmit={submit}>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="e.g. How many active shops are there in each city?"
        rows={2}
        autoFocus
      />
      <div className="ask-bar">
        <label className="as-of">
          Today is
          <input type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} />
        </label>
        <button type="submit" disabled={busy || !question.trim()}>
          {busy ? "Thinking…" : "Ask"}
        </button>
      </div>
    </form>
  );
}
