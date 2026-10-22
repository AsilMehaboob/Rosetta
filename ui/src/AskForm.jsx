import { useEffect, useRef, useState } from "react";

// The message box pinned to the bottom, as in a chat app. Enter sends, Shift+Enter adds a line.
export default function AskForm({ onAsk, busy }) {
  const [question, setQuestion] = useState("");
  const [asOf, setAsOf] = useState("");
  const box = useRef(null);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [question]);

  function submit(e) {
    e.preventDefault();
    if (!question.trim() || busy) return;
    onAsk(question.trim(), asOf);
    setQuestion("");
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) submit(e);
  }

  return (
    <form className="composer" onSubmit={submit}>
      <textarea
        ref={box}
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Ask Rosetta about your data…"
        rows={1}
        autoFocus
      />
      <div className="composer-bar">
        {/* "Today is" picker, hidden for now: answers use REFERENCE_DATE or the real today
        <label className="as-of" title="What 'today' means for words like 'last month'">
          Today is
          <input type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} />
        </label>
        */}
        <button type="submit" className="send" disabled={busy || !question.trim()} aria-label="Ask">
          ↑
        </button>
      </div>
    </form>
  );
}
