"use client";
import { useRef, useState } from "react";

export default function CodeBlock(props: React.ComponentProps<"pre">) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(ref.current?.innerText ?? "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  }
  return (
    <div className="code-wrap">
      <button className="copy-btn" onClick={copy}>{copied ? "Copied" : "Copy"}</button>
      <pre ref={ref} {...props} />
    </div>
  );
}
