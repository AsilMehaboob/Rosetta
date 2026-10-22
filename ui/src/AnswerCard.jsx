const STATUS = {
  answered: { text: "Answered", tone: "ok" },
  cannot_answer: { text: "Can't answer", tone: "warn" },
  failed: { text: "Failed", tone: "bad" },
};

export default function AnswerCard({ result }) {
  const status = STATUS[result.status] || { text: result.status, tone: "warn" };
  return (
    <section className="card">
      <div className="card-head">
        <span className={`badge ${status.tone}`}>{status.text}</span>
        <span className="question">{result.question}</span>
      </div>
      <p className="answer-text">{result.answer}</p>
      {result.explanation && <p className="explanation">{result.explanation}</p>}
      {result.notes?.length > 0 && (
        <ul className="notes">
          {result.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
