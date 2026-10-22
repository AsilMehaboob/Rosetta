import { Fragment, useCallback, useEffect, useState } from "react";
import { getBenchmark, getEvalStatus, getEvaluation, listEvaluations, startEvaluation } from "./api.js";
import ResultTable from "./ResultTable.jsx";
import SqlBlock from "./SqlBlock.jsx";

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

// Questions down, runs across: each cell fills in with its verdict as the run finishes.
function LiveGrid({ status }) {
  const ids = status.questions || [];
  const runs = status.runs_per_question || 0;
  if (!ids.length || !runs) return null;
  const done = new Map((status.completed || []).map((c) => [`${c.question}/${c.run}`, c.verdict]));
  return (
    <div className="live-grid" style={{ gridTemplateColumns: `auto repeat(${runs}, 18px)` }}>
      <span />
      {Array.from({ length: runs }, (_, r) => (
        <span key={r} className="live-run">
          {r + 1}
        </span>
      ))}
      {ids.map((id) => (
        <Fragment key={id}>
          <span className="live-id">{id}</span>
          {Array.from({ length: runs }, (_, r) => {
            const v = done.get(`${id}/${r + 1}`);
            const now = status.question === id && status.run === r + 1;
            const cls = v ? verdict(v).tone : now ? "now" : "todo";
            return <span key={r} className={`vdot ${cls}`} title={v ? `Run ${r + 1}: ${verdict(v).text}` : now ? "Asking now" : "Waiting"} />;
          })}
        </Fragment>
      ))}
    </div>
  );
}

function Dot({ v, run }) {
  const { text, tone } = verdict(v);
  return <span className={`vdot ${tone}`} title={`Run ${run}: ${text}`} />;
}

// The same comparable form the harness uses: numbers to 4 decimals, dates as YYYY-MM-DD, text trimmed.
const DAY = /^(\d{4}-\d{2}-\d{2})(?:[T ]00:00:00(?:\.0+)?)?$/;
const MONTH = /^(\d{4})-(\d{2})$/;
const MASKED = "***";

function norm(v) {
  if (v === null || v === undefined) return "null";
  if (typeof v === "boolean") return `b:${v}`;
  if (typeof v === "number") return `n:${Math.round(v * 1e4) / 1e4}`;
  const text = String(v).trim();
  const day = text.match(DAY);
  if (day) return `s:${day[1]}`;
  const month = text.match(MONTH);
  if (month) return `s:${month[1]}-${month[2]}-01`;
  return `s:${text}`;
}

const valuesOf = (rows, j) => new Set(rows.map((r) => norm(r[j])));
const allValues = (rows) => new Set(rows.flatMap((r) => r.map(norm)));

// Which columns were paired, and which cells on each side have no counterpart on the other side.
function differences(record, expected) {
  const got = { columns: record.columns || [], rows: record.rows || [] };
  const ref = { columns: expected.columns, rows: expected.rows };
  const paired = Object.entries(record.paired_columns || {})
    .map(([e, g]) => [ref.columns.indexOf(e), got.columns.indexOf(g)])
    .filter(([j, k]) => j >= 0 && k >= 0);
  const extra = new Set((record.extra_columns || []).map((c) => got.columns.indexOf(c)));
  const gotPair = new Map(paired.map(([j, k]) => [k, j]));
  const refPair = new Map(paired);
  const gotAll = allValues(got.rows);
  const refAll = allValues(ref.rows);
  const gotSets = new Map(paired.map(([j]) => [j, valuesOf(ref.rows, j)]));
  const refSets = new Map(paired.map(([j, k]) => [j, valuesOf(got.rows, k)]));

  const off = (value, pool) => (value === MASKED || pool.has(norm(value)) ? "" : "cell-off");
  return {
    paired: paired.map(([j, k]) => [ref.columns[j], got.columns[k]]),
    missing: ref.columns.filter((_, j) => !refPair.has(j)),
    extra: got.columns.filter((_, k) => extra.has(k)),
    got: {
      columnClass: (k) => (extra.has(k) ? "col-extra" : ""),
      cellClass: (i, k) => {
        if (extra.has(k)) return "cell-extra";
        const j = gotPair.get(k);
        return off(got.rows[i][k], j === undefined ? refAll : gotSets.get(j));
      },
    },
    ref: {
      columnClass: (j) => (refPair.has(j) ? "" : "col-missing"),
      cellClass: (i, j) => off(ref.rows[i][j], refPair.has(j) ? refSets.get(j) : gotAll),
    },
  };
}

function Pairing({ diff }) {
  if (!diff.paired.length && !diff.missing.length && !diff.extra.length) return null;
  return (
    <div className="pairing">
      {diff.paired.map(([e, g]) => (
        <span key={e} className="chip" title="Same values in both results">
          {e === g ? e : `${e} ↔ ${g}`}
        </span>
      ))}
      {diff.missing.map((c) => (
        <span key={c} className="chip missing" title="In the reference, not in Rosetta's result">
          {c} missing
        </span>
      ))}
      {diff.extra.map((c) => (
        <span key={c} className="chip extra" title="Added by Rosetta">
          + {c}
        </span>
      ))}
    </div>
  );
}

// One side of the comparison: a query and the rows it returned.
function Pane({ title, meta, sql, columns, rows, rowCount, marks, children }) {
  return (
    <div className="pane">
      <div className="pane-head">
        <span>{title}</span>
        {meta && <span className="muted">{meta}</span>}
      </div>
      {children}
      {sql ? <SqlBlock sql={sql} /> : <p className="muted">No query.</p>}
      {columns?.length > 0 && (
        <>
          <ResultTable columns={columns} rows={rows} cellClass={marks?.cellClass} columnClass={marks?.columnClass} />
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
  const diff = record?.columns?.length && expected ? differences(record, expected) : null;
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

      {diff && (
        <>
          <Pairing diff={diff} />
          <p className="muted small legend">
            <span className="cell-off swatch" /> value not found in the other result
          </p>
        </>
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
            marks={diff?.got}
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
            marks={diff?.ref}
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
  const [questions, setQuestions] = useState([]);
  const [only, setOnly] = useState(""); // "" = every question
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
    getBenchmark().then(setQuestions).catch(() => {});
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

  // every question, or just the ones given (a single question can be checked on its own)
  async function run(ids) {
    setError("");
    try {
      const count = Math.min(10, Math.max(1, Math.round(Number(runs)) || 2));
      setRuns(count);
      setStatus(await startEvaluation(count, ids?.length ? ids : only ? [only] : null, asOf));
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
        <div className="toolbar">
          <label className="field">
            Questions
            <select value={only} onChange={(e) => setOnly(e.target.value)} disabled={status.running}>
              <option value="">All {questions.length || ""}</option>
              {questions.map((q) => (
                <option key={q.id} value={q.id} title={q.question}>
                  {q.id} · {q.level}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
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
          <label className="field" title="What 'today' means for the questions; blank means the real today">
            Today is
            <input type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} disabled={status.running} />
          </label>
          <button type="button" className="primary" onClick={() => run()} disabled={status.running}>
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
          <LiveGrid status={status} />
        </div>
      )}

      {error && <p className="error">{error}</p>}

      {summaries.length === 0 && !status.running && (
        <div className="eval-empty">
          <h2>No evaluation yet</h2>
          <p className="muted">Run one to see how Rosetta does on the benchmark questions.</p>
          <ol>
            <li>
              <strong>Expected answers</strong> come from running each question's reference query on the database.
            </li>
            <li>
              <strong>Every question is asked several times</strong>, each time afresh, through the same pipeline as the
              chat.
            </li>
            <li>
              <strong>Answers are compared as data</strong>: column names, column order and row order don't matter.
              Questions the data can't answer must be refused.
            </li>
          </ol>
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
                  <th />
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
                      <td>
                        <button
                          type="button"
                          className="ghost small-btn"
                          title={`Ask ${p.id} again, ${runs} time(s)`}
                          disabled={status.running}
                          onClick={(e) => {
                            e.stopPropagation();
                            run([p.id]);
                          }}
                        >
                          Run
                        </button>
                      </td>
                    </tr>
                    {open === p.id && detail && (
                      <tr className="q-open">
                        <td colSpan={8}>
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
