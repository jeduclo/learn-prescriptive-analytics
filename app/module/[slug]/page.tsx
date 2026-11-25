import Link from "next/link";
import { notFound } from "next/navigation";
import { allLessons, dayTitle } from "@/lib/content";
import { MODULES, moduleBySlug } from "@/lib/modules";

export function generateStaticParams() {
  return MODULES.map((m) => ({ slug: m.slug }));
}

export default async function ModulePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const m = moduleBySlug(slug);
  if (!m) notFound();
  const lessons = allLessons();
  return (
    <div className="page" style={{ ["--c" as string]: m.colour }}>
      <div className="crumbs"><Link href="/">Overview</Link> / Module {m.n}</div>
      <div className="module-head">
        <div className="eyebrow">Module {m.n} · Days {m.n * 10 - 9}–{m.n * 10}</div>
        <h1>{m.title}</h1>
        <p className="lead">{m.goal}</p>
      </div>
      <div className="two-col">
        <section className="card">
          <h2>What you will learn</h2>
          <ul>{m.learn.map((l) => <li key={l}>{l}</li>)}</ul>
        </section>
        <section className="card">
          <h2>Tools</h2>
          <div className="chips">{m.tools.map((t) => <span key={t} className="chip">{t}</span>)}</div>
          <h2 style={{ marginTop: "1.2rem" }}>Module capstone</h2>
          <p>{m.capstone}</p>
        </section>
      </div>
      <h2>The ten days</h2>
      <div className="day-list">
        {m.days.map((_, i) => {
          const d = m.n * 10 - 9 + i;
          const l = lessons.get(d);
          return (
            <Link key={d} href={`/module/${m.slug}/day/${d}`} className={`day-card ${l ? "" : "soon"}`}>
              <div className="dc-num">Day {d}</div>
              <div className="dc-title">{dayTitle(d)}</div>
              <div className="dc-status">{l ? "Lesson ready" : "Notes soon"}</div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
