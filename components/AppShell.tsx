"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import Sidebar, { type SidebarData } from "./Sidebar";
import ThemeToggle from "./ThemeToggle";

export default function AppShell({ data, children }: { data: SidebarData; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);       // desktop sidebar
  const [drawer, setDrawer] = useState(false);  // phone drawer
  const pathname = usePathname();
  useEffect(() => setDrawer(false), [pathname]);
  function toggle() {
    if (window.matchMedia("(max-width: 900px)").matches) setDrawer((d) => !d);
    else setOpen((o) => !o);
  }
  return (
    <div className={`shell ${open ? "" : "sidebar-closed"} ${drawer ? "drawer-open" : ""}`}>
      <header className="topbar">
        <button className="icon-btn" onClick={toggle} aria-label="Toggle sidebar" title="Toggle sidebar">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></svg>
        </button>
        <Link href="/" className="brand">
          <span className="logo">◆</span>
          <span className="brand-title">{data.title}</span>
        </Link>
        <nav className="top-links">
          <Link href="/">Overview</Link>
        </nav>
        <ThemeToggle />
      </header>
      <aside className="sidebar"><Sidebar data={data} /></aside>
      <div className="scrim" onClick={() => setDrawer(false)} />
      <main className="main">{children}</main>
    </div>
  );
}
