const STATUS = {
  answered: null, // a normal answer needs no badge
  cannot_answer: { text: "Can't answer from this data", tone: "warn" },
  failed: { text: "Could not answer", tone: "bad" },
};

export default function AnswerCard({ result }) {
  const status = STATUS[result.status];
  return (
    <div className="answer">
      {status && <span className={`badge ${status.tone}`}>{status.text}</span>}
      <p className="answer-text">{result.answer}</p>
      {result.explanation && <p className="explanation">{result.explanation}</p>}
      {result.notes?.length > 0 && (
        <ul className="notes">
          {result.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
