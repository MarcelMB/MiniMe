import { createRoot } from "react-dom/client";

/** What a browser without OPFS (Browser storage) gets instead of the app. */
export function Unsupported() {
  return (
    <div className="page">
      <h1>This browser cannot run MiniMe</h1>
      <div className="bento">
        <section className="card span4" aria-label="Supported browsers">
          <p>
            MiniMe keeps the runs it computes in the browser's private storage (the Origin Private File System), and
            this browser does not offer it.
          </p>
          <p className="muted">Open this page in a recent Chrome, Edge, Firefox or Safari.</p>
        </section>
      </div>
    </div>
  );
}

/** Replaces the page with `Unsupported` once it has loaded: `main.tsx` has rendered the app into
 * `#root` by then (and is not ours to change), and the app's requests would only fail. */
export function showUnsupported(): void {
  const show = () => {
    document.body.innerHTML = '<div id="root"></div>';
    createRoot(document.getElementById("root")!).render(<Unsupported />);
  };
  if (document.readyState === "complete") show();
  else window.addEventListener("load", show, { once: true });
}
