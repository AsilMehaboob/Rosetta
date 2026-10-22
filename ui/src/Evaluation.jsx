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

// One of the three things the checkpoint asks for (correct, consistent, fast): a headline figure, a bar, details.
function ScoreGroup({ title, share, value, caption, rows }) {
  const tone = share === null || share === undefined ? "" : share >= 0.8 ? "good" : share >= 0.5 ? "warn" : "bad";
  return (
    <section className="score">
      <h3>{title}</h3>
      <div className="score-value">{value}</div>
      <div className={`score-bar ${tone}`}>
        <div style={{ width: `${Math.round((share ?? 0) * 100)}%` }} />
      </div>
      <p className="score-caption">{caption}</p>
      <dl>
        {rows.map(([label, v, hint]) => (
          <div key={label} title={hint}>
            <dt>{label}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function Scorecard({ summary }) {
  const o = summary.overall;
  const questions = summary.per_question;
  const solved = questions.filter((p) => p.runs > 0 && p.correct_runs === p.runs).length;
  const consistent = questions.filter((p) => p.consistent).length;
  const correctRuns = questions.reduce((n, p) => n + p.correct_runs, 0);
  return (
    <>
      <div className="headline">
        <span className="headline-big">
          {solved} of {questions.length}
        </span>
        <span className="headline-text">
          questions answered correctly in every run
          <span className="muted">
            {" "}
            · {correctRuns} of {o.runs} runs correct
          </span>
        </span>
      </div>
      <div className="scores">
        <ScoreGroup
          title="Correct"
          share={o.accuracy}
          value={pct(o.accuracy)}
          caption="of runs gave the expected data or an honest refusal"
          rows={[
            ["Answerable runs correct", pct(o.answer_accuracy)],
            ["Queries that ran", pct(o.query_success), "Answerable questions whose query ran without error"],
            ["Unanswerable refused", pct(o.refusal_accuracy)],
            ["Right rows, column missing", `${o.partial_runs} runs`],
            ["Wrong answers", pct(o.wrong_answers), "Runs that gave different data, or answered an unanswerable question"],
          ]}
        />
        <ScoreGroup
          title="Consistent"
          share={o.consistency}
          value={pct(o.consistency)}
          caption="of questions returned the same data in every run"
          rows={[
            ["Same data every run", `${consistent} of ${questions.length} questions`],
            ["Correct every run", pct(o.solved_every_run)],
            ["Runs per question", summary.runs_per_question],
          ]}
        />
        <ScoreGroup
          title="Fast"
          share={o.fast}
          value={secs(o.seconds.median)}
          caption={`median per answer · ${pct(o.fast)} within ${summary.fast_seconds}s`}
          rows={[
            ["Mean", secs(o.seconds.mean)],
            ["95th percentile", secs(o.seconds.p95)],
            ["Slowest", secs(o.seconds.max)],
            ["Model calls", o.llm_calls.toLocaleString()],
          ]}
        />
      </div>
    </>
  );
}

function Dot({ v, run }) {
  const { text, tone } = verdict(v);
  return <span className={`vdot ${tone}`} title={`Run ${run}: ${text}`} />;
}

// One side of the comparison: a query and the rows it returned.
function Pane({ title, meta, sql, columns, rows, rowCount, children }) {
  return (
    <div className="pane">
      <div className="pane-head">
        <span>{title}</span>
        {meta && <span className="muted">{meta}</span>}
      </div>
      {children}
      {sql ? <pre className="sql">{sql}</pre> : <p className="muted">No query.</p>}
      {columns?.length > 0 && (
        <>
          <ResultTable columns={columns} rows={rows} />
          {rowCount > rows.length && (
            <p className="muted small">First {rows.length} of {rowCount.toLocaleString()} rows saved.</p>
          )}
        </>
      )}
    </div>
  );
}

function QuestionDetail({ item, runs, expected }) {
  const [shown, setShown] = useState(runs[0]?.run);
  const record = runs.find((r) => r.run === shown) || runs[0];
  const v = record && verdict(record.verdict);
  return (
    <div className="q-detail">
      {item.checks && <p className="muted">Checks: {item.checks}</p>}
      {item.note && <p className="q-note">{item.note}</p>}

      <div className="run-tabs">
        {runs.map((r) => (
          <button key={r.run} type="button" className={r.run === record?.run ? "on" : ""} onClick={() => setShown(r.run)}>
            <span className={`vdot ${verdict(r.verdict).tone}`} />
            Run {r.run}
          </button>
        ))}
      </div>

      {record && (
        <p className="run-verdict">
          <span className={`badge ${v.tone}`}>{v.text}</span>
          {record.reason && <span className="muted"> {record.reason}</span>}
        </p>
      )}

      <div className="compare">
        {record ? (
          <Pane
            title={`Rosetta · run ${record.run}`}
            meta={`${record.row_count} rows · ${secs(record.seconds)} · ${record.llm_calls ?? 0} model calls`}
            sql={record.sql}
            columns={record.columns}
            rows={record.rows}
            rowCount={record.row_count}
          >
            {record.answer && <p className="run-answer">{record.answer}</p>}
          </Pane>
        ) : (
          <div className="pane muted">No runs.</div>
        )}
        {expected ? (
          <Pane
            title="Reference query"
            meta={`${expected.row_count} rows`}
            sql={expected.sql}
            columns={expected.columns}
            rows={expected.rows}
            rowCount={expected.row_count}
          />
        ) : (
          <div className="pane">
            <div className="pane-head">
              <span>Expected</span>
            </div>
            <p className="muted">An honest “the data cannot answer this”, with no query and no numbers.</p>
          </div>
        )}
      </div>
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
          <Scorecard summary={summary} />

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
