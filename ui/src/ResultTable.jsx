import { cell, label } from "./format.js";

export default function ResultTable({ columns, rows }) {
  if (!columns?.length) return null;
  return (
    <section className="results">
      <div className="results-head">
        <span>Results</span>
        <span className="muted">
          {rows.length.toLocaleString()} {rows.length === 1 ? "row" : "rows"}
        </span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c}>{label(c)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {row.map((v, j) => (
                  <td key={j} className={typeof v === "number" ? "num" : ""}>
                    {cell(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
