"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export type SidebarData = {
  title: string;
  written: number[];
  parts: { id: string; title: string; modules: { n: number; slug: string; short: string; colour: string; days: string[] }[] }[];
};

export default function Sidebar({ data }: { data: SidebarData }) {
  const pathname = usePathname();
  const written = new Set(data.written);
  const pct = Math.round((data.written.length / 80) * 100);
  const activeSlug = pathname.match(/^\/module\/([^/]+)/)?.[1];
  const activeDay = Number(pathname.match(/\/day\/(\d+)/)?.[1] ?? 0);
  return (
    <div className="side-inner">
      <div className="progress">
        <div className="progress-label"><span>Lessons published</span><span>{data.written.length} / 80</span></div>
        <div className="progress-bar"><div style={{ width: `${pct}%` }} /></div>
      </div>
      <Link href="/" className={`side-link ${pathname === "/" ? "active" : ""}`}>Overview</Link>
      <Link href="/course-data" className={`side-link ${pathname === "/course-data" ? "active" : ""}`}>Course data script</Link>
      {data.parts.map((p) => (
        <div key={p.id} className="side-part">
          <div className="side-part-title">{p.id} · {p.title}</div>
          {p.modules.map((m) => {
            const active = m.slug === activeSlug;
            return (
              <div key={m.n}>
                <Link href={`/module/${m.slug}`} className={`side-module ${active ? "active" : ""}`} style={{ ["--c" as string]: m.colour }}>
                  <span className="dot" />M{m.n} · {m.short}
                </Link>
                {active && (
                  <ol className="side-days">
                    {m.days.map((t, i) => {
                      const d = (m.n - 1) * 10 + i + 1;
                      const live = written.has(d);
                      return (
                        <li key={d} className={`${live ? "" : "dim"} ${d === activeDay ? "current" : ""}`}>
                          <Link href={`/module/${m.slug}/day/${d}`}>
                            <span className="tick">{live ? "✓" : "·"}</span>
                            <span className="dnum">{d}</span> {t}
                          </Link>
                        </li>
                      );
                    })}
                  </ol>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
