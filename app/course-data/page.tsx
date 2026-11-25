import Markdown from "@/components/Markdown";
import { dataScript } from "@/lib/content";

export const metadata = { title: "Course data script" };

export default function CourseData() {
  const code = dataScript();
  return (
    <div className="page">
      <div className="module-head">
        <div className="eyebrow">Course data</div>
        <h1>make_data.py</h1>
        <p className="lead">
          One script creates all the Northbound Fresh data. Copy it into the root of your repository as <code>make_data.py</code> and
          run <code>python make_data.py</code>. It uses a fixed seed, so everyone gets identical files. When later modules add data,
          this page shows the new version: copy it again and rerun.
        </p>
      </div>
      <Markdown text={"```python\n" + code + "\n```"} />
    </div>
  );
}
