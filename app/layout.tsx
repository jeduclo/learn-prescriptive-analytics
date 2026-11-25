import "katex/dist/katex.min.css";
import "./globals.css";
import type { Metadata } from "next";
import AppShell from "@/components/AppShell";
import { allLessons } from "@/lib/content";
import { COURSE, MODULES, PARTS } from "@/lib/modules";

export const metadata: Metadata = {
  title: COURSE.title,
  description: COURSE.subtitle,
};

// Sets the saved theme before the page paints, so there is no flash of the wrong theme.
const themeScript = `try{var t=localStorage.getItem("theme");document.documentElement.dataset.theme=t==="light"?"light":"dark"}catch(e){document.documentElement.dataset.theme="dark"}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const data = {
    title: COURSE.title,
    written: [...allLessons().keys()].sort((a, b) => a - b),
    parts: PARTS.map((p) => ({
      id: p.id,
      title: p.title,
      modules: p.modules.map((n) => {
        const m = MODULES[n - 1];
        return { n: m.n, slug: m.slug, short: m.short, colour: m.colour, days: m.days.map((t, i) => allLessons().get(m.n * 10 - 9 + i)?.title ?? t) };
      }),
    })),
  };
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <AppShell data={data}>{children}</AppShell>
      </body>
    </html>
  );
}
