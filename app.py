"""Tiny search UI: type a query, flip between systems, watch the order change.

    python app.py            # then open http://127.0.0.1:8000

No framework on purpose -- the interesting part of this project is retrieval,
not a web stack. Swap in Streamlit/Flask if you prefer.
"""

import argparse
import html
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if sys.prefix == os.path.dirname(sys.executable):
    import warnings
    warnings.warn(
        f"app.py is running under {sys.executable}, which is likely not the project "
        f"venv (.venv). Dense/rerank routes need sentence-transformers; restart with "
        f".venv/Scripts/python.exe to avoid 503s.",
        RuntimeWarning,
    )

import search as searcher

try:
    import rerank as reranker
except ImportError:  # sentence-transformers missing
    reranker = None

SYSTEMS = ["bm25", "dense", "hybrid", "hybrid+rerank"]

DEFAULT_SYSTEM = "hybrid+rerank" if reranker else "dense"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Minecraft semantic search</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 system-ui, sans-serif; max-width: 820px; margin: 3rem auto; padding: 0 1rem; }
  h1 { font-size: 1.3rem; margin-bottom: .25rem; }
  .sub { opacity: .65; margin-top: 0; font-size: .9rem; }
  form { display: flex; gap: .5rem; margin: 1.5rem 0; flex-wrap: wrap; }
  input[type=text] { flex: 1 1 320px; padding: .55rem .7rem; font-size: 1rem;
    border: 1px solid #8884; border-radius: 8px; background: transparent; color: inherit; }
  select, button { padding: .55rem .7rem; border-radius: 8px; border: 1px solid #8884;
    background: transparent; color: inherit; font-size: .95rem; cursor: pointer; }
  .hit { padding: .7rem 0; border-top: 1px solid #8882; }
  .hit .meta { font-size: .82rem; opacity: .6; }
  .rank { display: inline-block; min-width: 1.5rem; opacity: .5; font-variant-numeric: tabular-nums; }
  .snippet { margin: .2rem 0 0; }
  .hit a { color: inherit; text-decoration: underline; text-decoration-color: #8888; }
  .hit a:hover { text-decoration-color: currentColor; }
  .empty { opacity: .6; margin-top: 2rem; }
</style></head>
<body>
<h1>Minecraft semantic search</h1>
<p class="sub">BM25 vs. embeddings vs. hybrid fusion vs. cross-encoder reranking.</p>
<form id="f">
  <input id="q" type="text" placeholder="how do i make plants grow faster" autofocus>
  <select id="sys">""" + "".join(f"<option{' selected' if s == DEFAULT_SYSTEM else ''}>{s}</option>" for s in SYSTEMS) + """</select>
  <button type="submit">Search</button>
</form>
<div id="out"></div>
<script>
const f = document.getElementById('f'), q = document.getElementById('q'),
      sys = document.getElementById('sys'), out = document.getElementById('out');
async function go(e) {
  if (e) e.preventDefault();
  const term = q.value.trim();
  if (!term) return;
  out.innerHTML = '<p class="empty">searching...</p>';
  try {
    const res = await fetch('/api/search?q=' + encodeURIComponent(term) + '&system=' + encodeURIComponent(sys.value));
    const data = await res.json();
    if (data.error) { out.innerHTML = '<p class="empty"></p>'; out.firstChild.textContent = data.error; return; }
    if (!data.results.length) { out.innerHTML = '<p class="empty">no results</p>'; return; }
    out.innerHTML = data.results.map(r =>
      '<div class="hit"><div class="meta"><span class="rank">' + r.rank + '</span>' +
      '<strong><a href="' + r.url.replace(/"/g, '%22') + '" target="_blank" rel="noopener noreferrer">' +
      r.page + '</a></strong> &rsaquo; ' + r.section +
      ' &middot; ' + r.category + ' &middot; score ' + r.score.toFixed(4) + '</div>' +
      '<p class="snippet"></p></div>').join('');
    const snippets = out.querySelectorAll('.snippet');
    data.results.forEach((r, i) => { snippets[i].textContent = r.snippet; });
  } catch (err) {
    out.innerHTML = '<p class="empty"></p>'; out.firstChild.textContent = 'request failed: ' + err;
  }
}
f.addEventListener('submit', go);
sys.addEventListener('change', () => go());
</script>
</body></html>
"""


def run_query(query, system, k=10):
    if system == "hybrid+rerank":
        if reranker is None:
            raise RuntimeError("reranking needs sentence-transformers installed")
        return reranker.search_with_rerank(query, k=k, retrieve_k=20)
    return searcher.search(query, system=system, k=k)


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, status=200, content_type="application/json; charset=utf-8"):
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send(PAGE, content_type="text/html; charset=utf-8")
            return
        if parsed.path != "/api/search":
            self._send(json.dumps({"error": "not found"}), status=404)
            return

        params = urllib.parse.parse_qs(parsed.query)
        query = (params.get("q") or [""])[0]
        system_raw = (params.get("system") or [DEFAULT_SYSTEM])[0]
        if not query.strip():
            self._send(json.dumps({"error": "empty query"}))
            return
        system = system_raw.replace(" ", "+")
        try:
            results = run_query(query, system)
            payload = {
                "system": system,
                "results": [
                    {
                        "rank": r["rank"],
                        "page": html.unescape(r["page"]) and r["page"],
                        "section": r["section"],
                        "category": r["category"],
                        "url": r["url"],
                        "score": r["score"],
                        "snippet": " ".join(r["text"].split())[:300],
                    }
                    for r in results
                ],
            }
            self._send(json.dumps(payload))
        except Exception as exc:  # surface the reason in the UI rather than a blank page
            import traceback

            self._send(
                json.dumps(
                    {
                        "error": True,
                        "error_type": type(exc).__name__,
                        "error_message": (
                            str(exc)
                            if sys.version_info < (3,)
                            else f"{type(exc).__name__}: {exc}"
                        ),
                        "error_traceback": traceback.format_exc(),
                    }
                )
            )

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"serving on http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
