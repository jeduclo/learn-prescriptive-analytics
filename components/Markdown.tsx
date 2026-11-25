import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import CodeBlock from "./CodeBlock";

function textOf(node: React.ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (node && typeof node === "object" && "props" in node) return textOf((node as { props: { children?: React.ReactNode } }).props.children);
  return "";
}

function slugify(text: string) {
  return text.toLowerCase().replace(/[`*_]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

// Inline maths uses $$…$$ so that prices like "$17,500" stay plain text.
export default function Markdown({ text }: { text: string }) {
  return (
    <div className="prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: false }]]}
        rehypePlugins={[rehypeRaw, rehypeKatex, [rehypeHighlight, { detect: false, ignoreMissing: true }]]}
        components={{
          h2: ({ children }) => <h2 id={slugify(textOf(children))}>{children}</h2>,
          pre: (p) => <CodeBlock {...(p as React.ComponentProps<"pre">)} />,
          // eslint-disable-next-line @next/next/no-img-element
          img: (p) => <img {...p} alt={p.alt ?? ""} loading="lazy" draggable={false} />,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
