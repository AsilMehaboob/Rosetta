import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// Chart colours come from the CSS theme (light or dark), read when the theme changes.
const TOKENS = ["--chart-1", "--chart-2", "--chart-3", "--chart-4", "--muted", "--border", "--text", "--surface"];

function useTheme() {
  const read = () => {
    const css = getComputedStyle(document.documentElement);
    return Object.fromEntries(TOKENS.map((t) => [t, css.getPropertyValue(t).trim()]));
  };
  const [theme, setTheme] = useState(read);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setTheme(read());
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return theme;
}

const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
const full = (v) => (typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : v);

function xText(value, time) {
  if (value === null || value === undefined) return "No value";
  if (!time) return String(value);
  const m = String(value).match(/^(\d{4})-(\d{2})(?:-(\d{2}))?/);
  if (!m) return String(value);
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3] || 1));
  // a month start reads as the month ("Jul 2026"); any other day as the date
  return !m[3] || m[3] === "01"
    ? d.toLocaleDateString(undefined, { month: "short", year: "numeric" })
    : d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

function ChartTooltip({ active, payload, label, time }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip">
      <div className="chart-tip-title">{xText(label, time)}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="chart-tip-row">
          <span className="chart-swatch" style={{ background: p.color }} />
          <span>{p.name}</span>
          <strong>{full(p.value)}</strong>
          {p.payload.share !== undefined && <span className="muted">{p.payload.share}%</span>}
        </div>
      ))}
    </div>
  );
}

// bars grow and lines draw in, unless the viewer asked for less motion
const MOTION = !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
const ANIMATION = { isAnimationActive: MOTION, animationDuration: 900, animationEasing: "ease-out" };

export default function Chart({ chart }) {
  const theme = useTheme();
  if (!chart) return null;
  const colors = ["--chart-1", "--chart-2", "--chart-3", "--chart-4"].map((t) => theme[t]);
  const axis = { stroke: theme["--muted"], fontSize: 12, tickLine: false, axisLine: false };
  const grid = <CartesianGrid stroke={theme["--border"]} strokeDasharray="3 3" vertical={false} />;
  const time = chart.x.time;
  const tip = <Tooltip content={<ChartTooltip time={time} />} cursor={{ fill: theme["--border"], opacity: 0.35 }} />;
  const legend = chart.series.length > 1 && (
    <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, color: theme["--muted"], paddingTop: 8 }} />
  );

  let body;
  if (chart.type === "line") {
    body = (
      <LineChart data={chart.data} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
        {grid}
        <XAxis dataKey="x" {...axis} tickFormatter={(v) => xText(v, time)} />
        <YAxis {...axis} width={48} tickFormatter={compact.format} />
        {tip}
        {legend}
        {chart.series.map((s, i) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={colors[i % 4]} strokeWidth={2.5}
            dot={{ r: 3, fill: colors[i % 4] }} activeDot={{ r: 5 }} {...ANIMATION} animationDuration={1200} />
        ))}
      </LineChart>
    );
  } else if (chart.type === "column") {
    body = (
      <BarChart data={chart.data} margin={{ top: 20, right: 16, left: 0, bottom: 0 }} barCategoryGap="28%">
        {grid}
        <XAxis dataKey="x" {...axis} tickFormatter={(v) => xText(v, time)} />
        <YAxis {...axis} width={48} tickFormatter={compact.format} />
        {tip}
        {legend}
        {chart.series.map((s, i) => (
          <Bar key={s.key} dataKey={s.key} name={s.label} fill={colors[i % 4]} radius={[6, 6, 0, 0]} maxBarSize={56}
            {...ANIMATION} animationBegin={i * 150}>
            {chart.data.length <= 6 && (
              <LabelList dataKey={s.key} position="top" formatter={full} fill={theme["--muted"]} fontSize={11} />
            )}
          </Bar>
        ))}
      </BarChart>
    );
  } else {
    // a ranking or the parts of a whole: horizontal bars, biggest first
    const s = chart.series[0];
    const share = chart.type === "share";
    body = (
      <BarChart data={chart.data} layout="vertical" margin={{ top: 4, right: share ? 56 : 48, left: 8, bottom: 4 }}
        barCategoryGap="22%">
        <CartesianGrid stroke={theme["--border"]} strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" {...axis} tickFormatter={compact.format} />
        <YAxis type="category" dataKey="x" {...axis} width={140} tickFormatter={(v) => {
          const t = xText(v, time);
          return t.length > 20 ? `${t.slice(0, 19)}…` : t;
        }} />
        {tip}
        <Bar dataKey={s.key} name={s.label} radius={[0, 6, 6, 0]} maxBarSize={26} {...ANIMATION}>
          {chart.data.map((d, i) => (
            <Cell key={i} fill={colors[0]} fillOpacity={1 - (i / Math.max(chart.data.length, 1)) * 0.55} />
          ))}
          <LabelList dataKey={share ? "share" : s.key} position="right" fill={theme["--muted"]} fontSize={11}
            formatter={(v) => (share ? `${v}%` : full(v))} />
        </Bar>
      </BarChart>
    );
  }

  const height = chart.type === "bar" || chart.type === "share" ? Math.max(160, chart.data.length * 34 + 30) : 280;
  return (
    <section className="chart-card">
      <div className="chart-head">
        <span className="chart-title">{chart.title}</span>
        {chart.note && <span className="muted">{chart.note}</span>}
      </div>
      <div className="chart-body" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          {body}
        </ResponsiveContainer>
      </div>
      <p className="chart-reason">{chart.reason}</p>
    </section>
  );
}
