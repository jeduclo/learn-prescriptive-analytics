// Reads lesson notes from content/notes at build time and splits them into days.
// Notes format: "# DAY N — Title", a "| Time | Block |" table, then the lesson.
import fs from "node:fs";
import path from "node:path";
import { moduleOfDay } from "./modules";

export type Block = { start: number; end: number; label: string; kind: "break" | "commit" | "project" | "lesson" };

export type Lesson = {
  day: number;
  title: string;
  intro: string;
  plan: Block[];
  body: string;
  checklist: string[];
  sections: { id: string; text: string }[];
};

export function slugify(text: string) {
  return text.toLowerCase().replace(/[`*_]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

const NOTES_DIR = path.join(process.cwd(), "content", "notes");

function toMinutes(t: string) {
  const [h, m] = t.trim().split(":").map(Number);
  return h * 60 + m;
}

function kindOf(label: string): Block["kind"] {
  const l = label.toLowerCase();
  if (l.startsWith("break")) return "break";
  if (l.includes("commit")) return "commit";
  if (l.includes("mini-project") || l.includes("capstone")) return "project";
  return "lesson";
}

function parseDay(day: number, title: string, raw: string): Lesson {
  let text = raw.replace(/\n---\s*$/, "").trim();
  // The timetable: rows like "| 0:00 – 0:45 | 1.1 ... |"
  const plan: Block[] = [];
  const rowRe = /^\|\s*(\d+:\d\d)\s*[–-]\s*(\d+:\d\d)\s*\|\s*(.+?)\s*\|\s*$/gm;
  let m: RegExpExecArray | null;
  while ((m = rowRe.exec(text))) {
    plan.push({ start: toMinutes(m[1]), end: toMinutes(m[2]), label: m[3], kind: kindOf(m[3]) });
  }
  // Remove the timetable table (header, separator and rows) from the body: the page draws it as a chart
  text = text.replace(/^\| Time \| Block \|\n\|[-| ]+\|\n(?:\|.*\|\n?)+/m, "");
  const firstPara = text.split(/\n\s*\n/)[0] ?? "";
  const intro = firstPara.startsWith("#") || firstPara.startsWith("|") ? "" : firstPara;
  if (intro) text = text.slice(text.indexOf(firstPara) + firstPara.length);
  const checklist: string[] = [];
  const cl = text.match(/### Day \d+ checklist\n([\s\S]*)$/);
  if (cl) for (const line of cl[1].split("\n")) {
    const item = line.match(/^- \[[ x]\] (.+)$/);
    if (item) checklist.push(item[1]);
  }
  // Images in notes point at figures/…; the site serves them from /figures/…
  text = text.replace(/\]\(figures\//g, "](/figures/");
  const sections = [...text.matchAll(/^## (.+)$/gm)].map((h) => ({ id: slugify(h[1]), text: h[1].replace(/[`*]/g, "") }));
  return { day, title, intro, plan, body: text.trim(), checklist, sections };
}

let cache: Map<number, Lesson> | null = null;

export function allLessons(): Map<number, Lesson> {
  if (cache) return cache;
  const lessons = new Map<number, Lesson>();
  if (fs.existsSync(NOTES_DIR)) {
    for (const file of fs.readdirSync(NOTES_DIR).filter((f) => f.endsWith(".md")).sort()) {
      const md = fs.readFileSync(path.join(NOTES_DIR, file), "utf8").replace(/\r\n/g, "\n");
      const heads = [...md.matchAll(/^# DAY (\d+) — (.+)$/gm)];
      heads.forEach((h, i) => {
        const start = (h.index ?? 0) + h[0].length;
        const end = i + 1 < heads.length ? heads[i + 1].index : md.length;
        const day = Number(h[1]);
        lessons.set(day, parseDay(day, h[2].trim(), md.slice(start, end)));
      });
    }
  }
  cache = lessons;
  return lessons;
}

export function lesson(day: number) {
  return allLessons().get(day);
}

export function dayTitle(day: number) {
  return lesson(day)?.title ?? moduleOfDay(day).days[(day - 1) % 10];
}

export function writtenCount() {
  return allLessons().size;
}

export function dataScript() {
  return fs.readFileSync(path.join(process.cwd(), "content", "data", "make_data.py"), "utf8");
}
