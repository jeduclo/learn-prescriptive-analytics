import type { Block } from "@/lib/content";

const LABELS: Record<Block["kind"], string> = { lesson: "Lesson", project: "Project", break: "Break", commit: "Commit" };

function fmt(min: number) {
  return `${Math.floor(min / 60)}:${String(min % 60).padStart(2, "0")}`;
}

export default function TimePlan({ plan }: { plan: Block[] }) {
  if (!plan.length) return null;
  const total = Math.max(...plan.map((b) => b.end));
  return (
    <div className="timeplan">
      <div className="timeplan-head">
        <h3>Today&apos;s 5-hour plan</h3>
        <div className="legend">
          {(Object.keys(LABELS) as Block["kind"][]).map((k) => (
            <span key={k}><i className={`sw ${k}`} />{LABELS[k]}</span>
          ))}
        </div>
      </div>
      <div className="tp-rows">
        {plan.map((b, i) => (
          <div className="tp-row" key={i}>
            <div className="tp-time">{fmt(b.start)}–{fmt(b.end)}</div>
            <div className="tp-track">
              <div className={`tp-bar ${b.kind}`} style={{ left: `${(b.start / total) * 100}%`, width: `${((b.end - b.start) / total) * 100}%` }} />
            </div>
            <div className="tp-label">{b.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
