import { Fragment, useCallback, useEffect, useState } from "react";
import { getEvalStatus, getEvaluation, listEvaluations, startEvaluation } from "./api.js";
import ResultTable from "./ResultTable.jsx";

// How each verdict reads, and whether it counts as correct (good), nearly (warn) or wrong (bad).
const VERDICTS = {
  match: { text: "Match", tone: "good" },
  match_extra_columns: { text: "Match, extra columns", tone: "good" },
  correct_refusal: { text: "Correctly refused", tone: "good" },
  partial: { text: "Right rows, column missing", tone: "warn" },
  mismatch: { text: "Different data", tone: "bad" },
  wrong_refusal: { text: "Refused an answerable question", tone: "bad" },
  answered_unanswerable: { text: "Answered an unanswerable question", tone: "bad" },
  failed: { text: "No working query", tone: "bad" },
  error: { text: "Error", tone: "bad" },
};

const verdict = (v) => VERDICTS[v] || { text: v, tone: "bad" };
const pct = (x) => (x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`);
const secs = (x) => (x === null || x === undefined ? "—" : `${x.toLocaleString(undefined, { maximumFractionDigits: 1 })}s`);

function when(runId) {
  const m = runId.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/);
  if (!m) return runId;
  return new Date(`${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:00`).toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function Metric({ label, value, hint }) {
  return (
    <div className="metric" title={hint}>
      <span className="metric-value">{value}</span>
      <span className="metric-label">{label}</span>
    </div>
  );
}

function Dot({ v, run }) {
  const { text, tone } = verdict(v);
  return <span className={`vdot ${tone}`} title={`Run ${run}: ${text}`} />;
}

function RunDetail({ record }) {
  const { text, tone } = verdict(record.verdict);
  return (
    <div className="run">
      <div className="run-head">
        <span className={`badge ${tone}`}>Run {record.run} · {text}</span>
        <span className="muted">
          {secs(record.seconds)} · {record.llm_calls ?? 0} model calls · {record.row_count} rows
        </span>
      </div>
      {record.reason && <p className="run-reason">{record.reason}</p>}
      {record.answer && <p className="run-answer">{record.answer}</p>}
      {record.sql && (
        <details className="evidence">
          <summary>Query</summary>
          <pre className="sql">{record.sql}</pre>
        </details>
      )}
      {record.columns?.length > 0 && (
        <details className="evidence">
          <summary>Rows</summary>
          <ResultTable columns={record.columns} rows={record.rows} />
        </details>
      )}
    </div>
  );
}

function QuestionDetail({ item, runs, expected }) {
  return (
    <div className="q-detail">
      {item.checks && <p className="muted">Checks: {item.checks}</p>}
      {item.note && <p className="q-note">{item.note}</p>}
      {expected ? (
        <details className="evidence">
          <summary>Expected answer ({expected.row_count} rows, from the reference query)</summary>
          <pre className="sql">{expected.sql}</pre>
          <ResultTable columns={expected.columns} rows={expected.rows} />
        </details>
      ) : (
        <p className="muted">Expected: an honest “the data cannot answer this”.</p>
      )}
      {runs.map((r) => (
        <RunDetail key={r.run} record={r} />
      ))}
    </div>
  );
}

export default function Evaluation() {
  const [summaries, setSummaries] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [status, setStatus] = useState({ running: false });
  const [open, setOpen] = useState(null);
  const [runs, setRuns] = useState(2);
  const [asOf, setAsOf] = useState("");
  const [error, setError] = useState("");

  const refresh = useCallback(async (pickNewest) => {
    try {
      const list = await listEvaluations();
      setSummaries(list);
      if (list.length) setSelected((s) => (pickNewest || !s ? list[0].run_id : s));
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh(false);
    getEvalStatus().then(setStatus).catch(() => {});
  }, [refresh]);

  // follow a running evaluation, then show its results
  useEffect(() => {
    if (!status.running) return undefined;
    const timer = setInterval(async () => {
      try {
        const next = await getEvalStatus();
        setStatus(next);
        if (!next.running) {
          if (next.error) setError(next.error);
          refresh(true);
        }
      } catch (e) {
        setError(e.message);
      }
    }, 3000);
    return () => clearInterval(timer);
  }, [status.running, refresh]);

  useEffect(() => {
    if (!selected) return;
    setDetail(null);
    setOpen(null);
    getEvaluation(selected).then(setDetail).catch((e) => setError(e.message));
  }, [selected]);

  async function run() {
    setError("");
    try {
      const count = Math.min(10, Math.max(1, Math.round(Number(runs)) || 2));
      setRuns(count);
      setStatus(await startEvaluation(count, null, asOf));
    } catch (e) {
      setError(e.message);
    }
  }

  const summary = detail?.summary;
  const o = summary?.overall;

  return (
    <main className="eval">
      <div className="eval-head">
        <div>
          <h1>Evaluation</h1>
          <p className="muted">
            Each benchmark question is asked several times; the answers are compared with the expected data, not the
            query text.
          </p>
        </div>
        <div className="eval-controls">
          <label className="as-of">
            Runs
            <input
              type="number"
              className="runs"
              min={1}
              max={10}
              value={runs}
              onChange={(e) => setRuns(e.target.value)}
              disabled={status.running}
            />
          </label>
          <label className="as-of" title="What 'today' means for the questions">
            Today is
            <input type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} disabled={status.running} />
          </label>
          <button type="button" className="primary" onClick={run} disabled={status.running}>
            {status.running ? "Running…" : "Run evaluation"}
          </button>
        </div>
      </div>

      {status.running && (
        <div className="progress">
          <div className="progress-text">
            <span className="dot" />
            {status.question ? `Asking ${status.question}, run ${status.run}` : "Preparing the expected answers"}
            {status.total ? ` · ${status.done} of ${status.total} done` : ""}
            {status.last && <span className="muted"> · last: {status.last}</span>}
          </div>
          <div className="bar">
            <div style={{ width: `${status.total ? (status.done / status.total) * 100 : 2}%` }} />
          </div>
        </div>
      )}

      {error && <p className="error">{error}</p>}

      {summaries.length === 0 && !status.running && (
        <div className="empty eval-empty">
          <p>No evaluation yet. Run one to see how many answers are correct, consistent and fast.</p>
        </div>
      )}

      {summaries.length > 0 && (
        <div className="eval-meta">
          <select value={selected || ""} onChange={(e) => setSelected(e.target.value)}>
            {summaries.map((s) => (
              <option key={s.run_id} value={s.run_id}>
                {when(s.run_id)} · {pct(s.overall.accuracy)} correct
              </option>
            ))}
          </select>
          {summary && (
            <span className="muted">
              {summary.model} · layer {summary.layer_version} · today = {summary.reference_date} ·{" "}
              {summary.runs_per_question} runs per question
            </span>
          )}
        </div>
      )}

      {summary?.stopped && <p className="error">Stopped early: {summary.stopped}</p>}

      {o && (
        <>
          <div className="metrics">
            <Metric label="Correct runs" value={pct(o.accuracy)} hint={`${o.runs} runs in total`} />
            <Metric label="Correct every run" value={pct(o.solved_every_run)} hint="Questions right in all runs" />
            <Metric label="Consistent" value={pct(o.consistency)} hint="Questions with the same data in every run" />
            <Metric label="Queries ran" value={pct(o.query_success)} hint="Answerable questions whose query ran" />
            <Metric label="Refused correctly" value={pct(o.refusal_accuracy)} hint="Unanswerable questions refused" />
            <Metric label="Wrong answers" value={pct(o.wrong_answers)} hint="Runs that gave a wrong number" />
            <Metric label={`Within ${summary.fast_seconds}s`} value={pct(o.fast)} hint="Runs fast enough" />
            <Metric label="Median time" value={secs(o.seconds.median)} hint={`p95 ${secs(o.seconds.p95)} · max ${secs(o.seconds.max)}`} />
          </div>

          <div className="table-wrap eval-table">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Level</th>
                  <th>Question</th>
                  <th>Runs</th>
                  <th>Correct</th>
                  <th>Consistent</th>
                  <th>Median</th>
                </tr>
              </thead>
              <tbody>
                {summary.per_question.map((p) => (
                  <Fragment key={p.id}>
                    <tr className="q-row" onClick={() => setOpen(open === p.id ? null : p.id)}>
                      <td>{p.id}</td>
                      <td className="muted">{p.level}</td>
                      <td className="q-text">{p.question}</td>
                      <td>
                        {p.verdicts.map((v, i) => (
                          <Dot key={i} v={v} run={i + 1} />
                        ))}
                      </td>
                      <td className="num">
                        {p.correct_runs}/{p.runs}
                      </td>
                      <td>{p.consistent ? "Yes" : "No"}</td>
                      <td className="num">{secs(p.seconds.median)}</td>
                    </tr>
                    {open === p.id && detail && (
                      <tr className="q-open">
                        <td colSpan={7}>
                          <QuestionDetail
                            item={p}
                            runs={detail.runs.filter((r) => r.question_id === p.id)}
                            expected={detail.expected[p.id]}
                          />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
          <p className="fine">Click a question to see each run's verdict, query and rows next to the expected answer.</p>
        </>
      )}
    </main>
  );
}
