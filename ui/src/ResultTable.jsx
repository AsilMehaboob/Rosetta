import { cell, label } from "./format.js";

// cellClass(row, column) and columnClass(column) optionally mark cells and headers, e.g. to show differences.
export default function ResultTable({ columns, rows, title = "Results", cellClass, columnClass }) {
  if (!columns?.length) return null;
  return (
    <section className="results">
      <div className="results-head">
        <span>{title}</span>
        <span className="muted">
          {rows.length.toLocaleString()} {rows.length === 1 ? "row" : "rows"}
        </span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {columns.map((c, j) => (
                <th key={c} className={columnClass?.(j) || ""} title={c}>
                  {label(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {row.map((v, j) => (
                  <td key={j} className={`${typeof v === "number" ? "num" : ""} ${cellClass?.(i, j) || ""}`.trim()}>
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
