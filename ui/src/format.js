// Display helpers: friendly column headers and readable cell values.

export function label(column) {
  const words = column.toLowerCase().replace(/_/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

const MIDNIGHT = /^(\d{4})-(\d{2})-(\d{2})T00:00:00$/;
const STAMP = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/;

export function cell(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return value.toLocaleString();
  if (typeof value === "string") {
    const day = value.match(MIDNIGHT);
    if (day) return new Date(`${day[1]}-${day[2]}-${day[3]}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
    if (STAMP.test(value)) return new Date(value).toLocaleString();
  }
  return String(value);
}
