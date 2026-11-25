# Prescriptive Analytics & Decision Intelligence — course site

A Next.js site for the 80-day course: an overview page, one page per module, and one page per day (video, lesson contents, a chart of the 5-hour plan, and the full lesson). Days without notes show "Notes soon". There are no downloads: notes and the data script are read when the site is built and shown as pages, never served as files.

Dark mode is the default; the button at the top right switches to light mode and remembers the choice.

## 1. Run it on your computer (Windows, VS Code)

1. Install [Node.js](https://nodejs.org) 22 LTS or newer, [Git](https://git-scm.com/download/win) and [VS Code](https://code.visualstudio.com).
2. Unzip this folder, then in VS Code choose **File → Open Folder…** and open it.
3. Open a terminal (**Terminal → New Terminal**) and run:

```powershell
npm install
npm run dev
```

4. Open http://localhost:3000. Pages reload as you edit. Stop the server with `Ctrl+C`.

To check a production build before deploying: `npm run build`, then `npm start`.

## 2. Add a video

All 80 video slots are in `content/videos.ts`. Paste a YouTube link next to its day:

```ts
  1: "https://www.youtube.com/watch?v=XXXXXXXXXXX",
```

Until a link is there, the day shows "Video for Day N coming soon". The home page counts published videos.

## 3. Publish new lessons

1. Put the Markdown notes in `content/notes/`, named like `Module1_Days04-07_Your_First_Pyomo_Models.md`. One file can hold several days.
2. Put the notes' images in `public/figures/` (notes refer to them as `figures/name.png`).
3. If the data script changed, replace `content/data/make_data.py`.
4. Commit and push (section 4). Vercel rebuilds, and those days switch from "Notes soon" to full lessons.

The site relies on four patterns in the notes:

- each day starts with `# DAY N — Title` (an em dash);
- the first paragraph after it is the day's introduction;
- the timetable is a table headed `| Time | Block |` with rows like `| 0:00 – 0:45 | … |` (drawn as the plan chart);
- `## …` headings become the "In this lesson" list.

Maths: display formulas go between `$$` lines; inline maths uses `$$…$$` too, so prices like $17,500 stay plain text.

## 4. Put it on GitHub

1. Create an empty **private** repository on GitHub (no README), for example `prescriptive-analytics-site`.
2. In the VS Code terminal, in this folder:

```powershell
git init
git add .
git commit -m "Course site"
git branch -M main
git remote add origin https://github.com/YOUR-USER/prescriptive-analytics-site.git
git push -u origin main
```

VS Code asks you to sign in to GitHub the first time. Afterwards, use the **Source Control** panel (or `git add . ; git commit -m "…" ; git push`) for each update.

## 5. Deploy on Vercel

1. Sign in at [vercel.com](https://vercel.com) with your GitHub account.
2. **Add New… → Project**, pick the repository, and click **Deploy**. Vercel detects Next.js; no settings are needed.
3. Every push to `main` redeploys the site automatically.

## 6. Where things live

| What | Where |
|---|---|
| Course title, parts, modules, day titles, goals, tools, capstones | `lib/modules.ts` |
| Lesson notes | `content/notes/*.md` |
| Lesson images | `public/figures/*.png` |
| Data script shown on the "Course data script" page | `content/data/make_data.py` |
| YouTube links | `content/videos.ts` |
| Notes parsing (days, plan, contents) | `lib/content.ts` |
| Layout, sidebar, theme button | `components/AppShell.tsx`, `components/Sidebar.tsx`, `components/ThemeToggle.tsx` |
| Colours and styles (dark and light) | `app/globals.css` |
