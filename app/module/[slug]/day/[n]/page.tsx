import Link from "next/link";
import { notFound } from "next/navigation";
import Markdown from "@/components/Markdown";
import TimePlan from "@/components/TimePlan";
import Video from "@/components/Video";
import { dayTitle, lesson } from "@/lib/content";
import { MODULES, moduleBySlug } from "@/lib/modules";
import { VIDEOS } from "@/content/videos";

export function generateStaticParams() {
  return MODULES.flatMap((m) => m.days.map((_, i) => ({ slug: m.slug, n: String(m.n * 10 - 9 + i) })));
}

export async function generateMetadata({ params }: { params: Promise<{ n: string }> }) {
  const { n } = await params;
  return { title: `Day ${n}: ${dayTitle(Number(n))}` };
}

export default async function DayPage({ params }: { params: Promise<{ slug: string; n: string }> }) {
  const { slug, n } = await params;
  const m = moduleBySlug(slug);
  const d = Number(n);
  if (!m || !(d >= m.n * 10 - 9 && d <= m.n * 10)) notFound();
  const l = lesson(d);
  const prev = d > 1 ? d - 1 : null;
  const next = d < 80 ? d + 1 : null;
  const slugOf = (x: number) => MODULES[Math.floor((x - 1) / 10)].slug;
  return (
    <div className="page" style={{ ["--c" as string]: m.colour }}>
      <div className="crumbs"><Link href="/">Overview</Link> / <Link href={`/module/${m.slug}`}>Module {m.n}</Link> / Day {d}</div>
      <div className="module-head">
        <div className="eyebrow">Module {m.n} · {m.title} · Day {d} of 80</div>
        <h1>{dayTitle(d)}</h1>
        {l?.intro && <div className="lead"><Markdown text={l.intro} /></div>}
      </div>
      <Video url={VIDEOS[d]} day={d} />
      {l ? (
        <>
          {l.sections.length > 0 && (
            <section className="card toc">
              <h2>In this lesson</h2>
              <ul>{l.sections.map((sec) => <li key={sec.id}><a href={`#${sec.id}`}>{sec.text}</a></li>)}</ul>
            </section>
          )}
          <TimePlan plan={l.plan} />
          <article><Markdown text={l.body} /></article>
        </>
      ) : (
        <section className="card soon-card">
          <h2>Notes soon</h2>
          <p>This lesson is planned as part of Module {m.n}: <b>{m.title}</b>. Its notes will appear here when they are published.</p>
          <p className="muted">Module goal: {m.goal}</p>
        </section>
      )}
      <nav className="pager">
        {prev ? <Link href={`/module/${slugOf(prev)}/day/${prev}`}>← Day {prev}: {dayTitle(prev)}</Link> : <span />}
        {next ? <Link href={`/module/${slugOf(next)}/day/${next}`}>Day {next}: {dayTitle(next)} →</Link> : <span />}
      </nav>
    </div>
  );
}
