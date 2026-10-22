function Section({ title, children }) {
  return (
    <div className="ev-section">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

export default function Evidence({ result }) {
  const { sql, definitions = [], rules = [], values = [], dates = [] } = result;
  return (
    <details className="evidence">
      <summary>How was this calculated?</summary>

      {dates.length > 0 && (
        <Section title="How dates were read">
          <ul>
            {dates.map((d) => (
              <li key={d}>{d}</li>
            ))}
          </ul>
        </Section>
      )}

      {definitions.length > 0 && (
        <Section title="Definitions used">
          <ul>
            {definitions.map((d) => (
              <li key={d.term + d.table}>
                <strong>{d.term}</strong> — {d.definition}
                <span className="proof">verified: {d.proof}</span>
              </li>
            ))}
          </ul>
        </Section>
      )}

      {(rules.length > 0 || values.length > 0) && (
        <Section title="Rules and values applied">
          <ul>
            {rules.map((r) => (
              <li key={r.table + r.rule}>
                {r.rule} <span className="muted">({r.table})</span>
              </li>
            ))}
            {values.map((v) => (
              <li key={v.column + v.stored_value}>
                “{v.phrase}” matched the stored value <code>{v.stored_value}</code>{" "}
                <span className="muted">({v.column})</span>
              </li>
            ))}
          </ul>
        </Section>
      )}

      {sql && (
        <Section title="Query">
          <pre className="sql">{sql}</pre>
        </Section>
      )}

      <p className="meta">
        {[result.model, result.layer_version && `layer ${result.layer_version}`, result.seconds && `${result.seconds}s`,
          result.reference_date && `today = ${result.reference_date}`]
          .filter(Boolean)
          .join(" · ")}
      </p>
    </details>
  );
}
