import Link from "next/link";
import { allLessons, writtenCount } from "@/lib/content";
import { COURSE, MODULES, PARTS } from "@/lib/modules";
import { VIDEOS } from "@/content/videos";

export default function Home() {
  const written = writtenCount();
  const videos = Object.values(VIDEOS).filter(Boolean).length;
  const lessons = allLessons();
  return (
    <div className="page">
      <section className="hero">
        <div className="eyebrow">80-day specialty course</div>
        <h1>{COURSE.title}</h1>
        <p className="lead">{COURSE.subtitle}</p>
        <div className="stats">
          <div><b>{COURSE.days}</b><span>days</span></div>
          <div><b>8</b><span>modules</span></div>
          <div><b>{COURSE.hoursPerDay} h</b><span>per day</span></div>
          <div><b>{written}/80</b><span>lessons written</span></div>
          <div><b>{videos}/80</b><span>videos published</span></div>
        </div>
      </section>

      <section className="card">
        <h2>What you will be able to do</h2>
        <p>
          Most data work stops at describing what happened or predicting what will happen. This course owns the last step:
          computing the best action under real-world rules and uncertainty. Forecasts and causal estimates go in; executable,
          explainable decisions come out.
        </p>
        <p>
          You work on one company for 80 days: <b>Northbound Fresh</b>, a fictional Canadian grocery chain with 40 stores,
          3 distribution centres, a bakery, a truck fleet and a carbon target. Its data come from one seeded script, so every
          student gets the same numbers and every model is checked against a known answer.
        </p>
      </section>

      {PARTS.map((p) => (
        <section key={p.id}>
          <h2 className="part-title">Part {p.id} · {p.title}</h2>
          <div className="module-grid">
            {p.modules.map((n) => {
              const m = MODULES[n - 1];
              const done = m.days.filter((_, i) => lessons.has(m.n * 10 - 9 + i)).length;
              return (
                <Link key={m.n} href={`/module/${m.slug}`} className="module-card" style={{ ["--c" as string]: m.colour }}>
                  <div className="mc-top"><span className="mc-num">Module {m.n}</span><span className="mc-days">Days {m.n * 10 - 9}–{m.n * 10}</span></div>
                  <h3>{m.title}</h3>
                  <p>{m.goal}</p>
                  <div className="mc-progress"><div style={{ width: `${done * 10}%` }} /></div>
                  <div className="mc-foot">{done}/10 lessons written</div>
                </Link>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
