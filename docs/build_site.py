"""
Build the GitHub Pages landing page  docs/index.html  (light theme).

Composes the generated figures (docs/images) and tables (docs/tables) into a
single, self-contained, responsive page. Run after generate_assets.py:

    python docs/build_site.py
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "tables")


def table(name: str) -> str:
    p = os.path.join(TAB, name + ".html")
    with open(p, encoding="utf-8") as f:
        return f.read()


CSS = """
:root{
  --bg:#f7f9fc; --card:#ffffff; --ink:#1f2328; --muted:#57606a;
  --accent:#2563eb; --accent2:#0ea5e9; --line:#e3e8ef; --good:#15803d; --warn:#b45309;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:'Segoe UI',system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif;
  line-height:1.65;}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1060px;margin:0 auto;padding:0 22px}
header.hero{background:linear-gradient(135deg,#eef4ff 0%,#f7fbff 55%,#ffffff 100%);
  border-bottom:1px solid var(--line);padding:64px 0 46px;text-align:center}
.hero h1{font-size:2.9rem;margin:.1em 0 .15em;letter-spacing:-.5px;
  background:linear-gradient(90deg,#1d4ed8,#0ea5e9);-webkit-background-clip:text;
  background-clip:text;color:transparent}
.hero p.tag{font-size:1.18rem;color:var(--muted);margin:.2em auto 1.1em;max-width:680px}
.badges img{margin:3px;vertical-align:middle}
.cta{margin-top:18px}
.btn{display:inline-block;margin:6px;padding:11px 20px;border-radius:10px;
  font-weight:600;border:1px solid var(--line);background:#fff;color:var(--ink)}
.btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.btn:hover{text-decoration:none;transform:translateY(-1px);box-shadow:0 6px 18px rgba(37,99,235,.15)}
code.inline{background:#eef2ff;color:#1e40af;padding:2px 8px;border-radius:6px;
  font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.95em}
nav.toc{position:sticky;top:0;z-index:5;background:rgba(247,249,252,.92);
  backdrop-filter:blur(6px);border-bottom:1px solid var(--line);padding:10px 0}
nav.toc .wrap{display:flex;flex-wrap:wrap;gap:6px;justify-content:center}
nav.toc a{padding:6px 12px;border-radius:8px;color:var(--muted);font-size:.92rem;font-weight:600}
nav.toc a:hover{background:#eef2ff;color:var(--accent);text-decoration:none}
section{padding:46px 0;border-bottom:1px solid var(--line)}
section h2{font-size:1.8rem;margin:0 0 .2em}
.lead{color:var(--muted);max-width:780px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;
  padding:22px;margin:18px 0;box-shadow:0 1px 3px rgba(16,24,40,.04)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
.feature{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
.feature h3{margin:.1em 0 .35em;font-size:1.05rem}
.feature p{margin:0;color:var(--muted);font-size:.95rem}
.kbd{font-family:ui-monospace,Consolas,monospace}
figure{margin:14px 0;text-align:center}
figure img{max-width:100%;border:1px solid var(--line);border-radius:12px;background:#fff}
figcaption{color:var(--muted);font-size:.88rem;margin-top:6px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
@media(max-width:760px){.two{grid-template-columns:1fr}.hero h1{font-size:2.1rem}}
.tablebox{overflow-x:auto}
.math{background:#f5f8ff;border-left:4px solid var(--accent);border-radius:8px;
  padding:10px 16px;margin:12px 0;color:#0f254e;font-family:ui-monospace,Consolas,monospace;
  overflow-x:auto;font-size:.92rem}
.pill{display:inline-block;background:#eef2ff;color:#1e40af;border-radius:999px;
  padding:3px 12px;font-size:.8rem;font-weight:700;margin-bottom:8px}
pre.code{background:#0f172a00;border:1px solid var(--line);border-radius:12px;
  background:#f6f8fc;padding:16px;overflow-x:auto;font-size:.9rem;
  font-family:ui-monospace,Consolas,monospace;color:#0f172a}
.swatch img{width:100%;border-radius:10px;border:1px solid var(--line)}
footer{padding:40px 0 60px;text-align:center;color:var(--muted)}
footer a{font-weight:600}
.author{font-weight:700;color:var(--ink);font-size:1.05rem}
table{margin:0 auto}
"""

GH = "https://github.com/merwanroudane/pybayescointur"
PYPI = "https://pypi.org/project/pybayescointur/"

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pybayescointur — Bayesian panel unit-root &amp; cointegration tests</title>
<meta name="description" content="Bayesian unit-root and cointegration tests for panel data in Python, by Dr Merwan Roudane.">
<style>{CSS}</style>
</head>
<body>

<header class="hero">
  <div class="wrap">
    <h1>pybayescointur</h1>
    <p class="tag">Bayesian unit-root &amp; cointegration tests for <b>panel data</b> —
       four published methodologies, one consistent API, publication-quality
       tables &amp; visualisations.</p>
    <div class="badges">
      <a href="{PYPI}"><img alt="PyPI" src="https://img.shields.io/pypi/v/pybayescointur.svg?color=2563eb"></a>
      <a href="{PYPI}"><img alt="Python" src="https://img.shields.io/pypi/pyversions/pybayescointur.svg"></a>
      <a href="{GH}/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-15803d"></a>
      <img alt="colormap" src="https://img.shields.io/badge/colormap-Parula-f59e0b">
    </div>
    <div class="cta">
      <a class="btn primary" href="{PYPI}">📦 Install from PyPI</a>
      <a class="btn" href="{GH}">🐙 GitHub</a>
      <a class="btn" href="notebook/index.html">📓 Tutorial notebook</a>
    </div>
    <p style="margin-top:18px"><code class="inline">pip install pybayescointur</code></p>
  </div>
</header>

<nav class="toc"><div class="wrap">
  <a href="#about">Overview</a>
  <a href="#theory">Theory</a>
  <a href="#m1">Unit root (POR)</a>
  <a href="#m2">Structural break</a>
  <a href="#m3">Cross-section dep.</a>
  <a href="#m4">Cointegration</a>
  <a href="#palette">Palette</a>
  <a href="#author">Author</a>
</div></nav>

<section id="about"><div class="wrap">
  <span class="pill">OVERVIEW</span>
  <h2>What it does</h2>
  <p class="lead">Where the classical panel toolbox relies on asymptotics and a
   sharp null, <b>pybayescointur</b> integrates nuisance parameters out, treats
   the autoregressive root (and the cointegrating rank) as random, and returns
   directly interpretable <b>posterior odds ratios</b>, <b>model probabilities</b>
   and <b>Bayes factors</b>.</p>
  <div class="grid">
    <div class="feature"><h3>1 · Panel POR unit root</h3>
      <p>Posterior odds ratio with linear trend or augmentation terms.
         <br><i>Kumar, Chaturvedi &amp; Afifa (2016)</i></p></div>
    <div class="feature"><h3>2 · Structural break</h3>
      <p>Break in mean &amp; variance — 8 hypotheses, 14 posterior odds ratios.
         <br><i>Kumar &amp; Agiwal (2019)</i></p></div>
    <div class="feature"><h3>3 · Cross-sectional dependence</h3>
      <p>Marginal-likelihood comparison of 8 competing models.
         <br><i>Meligkotsidou, Tzavalis &amp; Vrontos</i></p></div>
    <div class="feature"><h3>4 · Panel cointegration</h3>
      <p>VECM with Savage–Dickey Bayes factors for the rank.
         <br><i>Koop, Leon-Gonzalez &amp; Strachan (2006)</i></p></div>
  </div>
</div></section>

<section id="theory"><div class="wrap">
  <span class="pill">THEORY</span>
  <h2>A little theory</h2>
  <p class="lead">For a null H₀ against an alternative H₁, the posterior odds
   ratio factorises into the prior odds times the Bayes factor:</p>
  <div class="math">&beta;<sub>01</sub> = [ P(H&#8320;) / P(H&#8321;) ] &times; [ P(y | H&#8320;) / P(y | H&#8321;) ] = P(H&#8320; | y) / P(H&#8321; | y)</div>
  <p class="lead">The Bayes factor integrates each model's parameters out of the
   likelihood, which automatically penalises complexity (an Occam factor):</p>
  <div class="math">P(y | H&#8342;) = &#8747; P(y | &theta;&#8342;, H&#8342;) &#183; p(&theta;&#8342; | H&#8342;) d&theta;&#8342;</div>
  <p class="lead"><b>Decision rule:</b> &beta;<sub>01</sub> &lt; 1 is evidence
   <i>against</i> the null (e.g. reject a unit root → trend stationary);
   &beta;<sub>01</sub> &gt; 1 favours it. Pooling the cross-section (n units) with
   the time dimension (T) sharply raises the power to tell a unit root apart from
   a near-unit root. All marginal likelihoods are evaluated in <b>log-space</b> to
   avoid overflow.</p>
</div></section>

<section id="m1"><div class="wrap">
  <span class="pill">METHOD 1</span>
  <h2>Panel unit root via the Posterior Odds Ratio</h2>
  <p class="lead">Model <span class="kbd">y&#7522;&#8348; = &mu;&#7522; + &delta;&#7522;t + u&#7522;&#8348;</span>,
   <span class="kbd">u&#7522;&#8348; = &rho; u&#7522;,&#8348;&#8331;&#8321; + &epsilon;&#7522;&#8348;</span>.
   Test H&#8320;: &rho;=1 (difference stationary) vs H&#8321;: a&lt;&rho;&lt;1 (trend stationary).</p>
  <pre class="code">res = pbc.bayesian_panel_unit_root(panel, model="augmented", k=2)
res.por          # &lt; 1  ->  reject the unit root</pre>
  <div class="two">
    <figure><img src="images/m1_panel.png" alt="panel series"><figcaption>Near-unit-root panel</figcaption></figure>
    <figure><img src="images/m1_por_sensitivity.png" alt="POR sensitivity"><figcaption>POR vs the prior P(H&#8320;)</figcaption></figure>
  </div>
  <div class="card tablebox">{table("m1_por")}</div>
</div></section>

<section id="m2"><div class="wrap">
  <span class="pill">METHOD 2</span>
  <h2>Unit root with a structural break in mean &amp; variance</h2>
  <p class="lead">A PAR(1) panel with a common break at T<sub>B</sub> that may shift
   both the mean and the error variance. Eight nested hypotheses produce 14
   posterior odds ratios.</p>
  <figure><img src="images/m2_panel.png" alt="break panel"><figcaption>Panel with a structural break at T<sub>B</sub> = 20</figcaption></figure>
  <figure><img src="images/m2_break_por.png" alt="break POR bars"><figcaption>The 14 posterior odds ratios (green = reject null)</figcaption></figure>
  <div class="card tablebox">{table("m2_break")}</div>
</div></section>

<section id="m3"><div class="wrap">
  <span class="pill">METHOD 3</span>
  <h2>Model comparison under cross-sectional dependence</h2>
  <p class="lead">Eight models — {{stationary, random walk}} × {{trend, no trend}} ×
   {{independent, cross-dependent}} — ranked by posterior probability. The
   estimated correlation matrix quantifies the co-movement across units.</p>
  <figure><img src="images/m3_panel.png" alt="G7 panel"><figcaption>G7-style log-GDP panel</figcaption></figure>
  <div class="two">
    <figure><img src="images/m3_probs.png" alt="model probabilities"><figcaption>Posterior model probabilities</figcaption></figure>
    <figure><img src="images/m3_corr.png" alt="correlation heatmap"><figcaption>Cross-sectional correlation (Parula)</figcaption></figure>
  </div>
  <div class="card tablebox">{table("m3_models")}</div>
  <div class="card tablebox">{table("m3_corr")}</div>
</div></section>

<section id="m4"><div class="wrap">
  <span class="pill">METHOD 4</span>
  <h2>Bayesian panel cointegration</h2>
  <p class="lead">Each unit has its own VECM with a possibly unit-specific
   cointegrating rank r&#7522;. Ranks are inferred via Savage–Dickey Bayes factors
   against the no-cointegration model.</p>
  <div class="two">
    <figure><img src="images/m4_panel.png" alt="cointegrated system"><figcaption>A cointegrated bivariate unit</figcaption></figure>
    <figure><img src="images/m4_rank.png" alt="rank posterior"><figcaption>Cointegrating-rank posterior</figcaption></figure>
  </div>
  <div class="card tablebox">{table("m4_rank")}</div>
</div></section>

<section id="palette"><div class="wrap">
  <span class="pill">DESIGN</span>
  <h2>Parula colour palette</h2>
  <p class="lead">Every heatmap defaults to the MATLAB <b>Parula</b> colormap,
   reproduced from its 64 RGB control points. Helpers:
   <code class="inline">parula_colors</code>,
   <code class="inline">matlab_jet_colors</code>,
   <code class="inline">turbo_colors</code>,
   <code class="inline">resolve_colorscale</code>.</p>
  <figure class="swatch"><img src="images/parula.png" alt="parula swatch"></figure>
</div></section>

<footer id="author"><div class="wrap">
  <p class="author">Dr Merwan Roudane</p>
  <p>
    ✉️ <a href="mailto:merwanroudane920@gmail.com">merwanroudane920@gmail.com</a> &nbsp;·&nbsp;
    📦 <a href="{PYPI}">PyPI</a> &nbsp;·&nbsp;
    🐙 <a href="{GH}">GitHub</a> &nbsp;·&nbsp;
    👤 <a href="https://github.com/merwanroudane">@merwanroudane</a>
  </p>
  <p style="margin-top:14px;font-size:.9rem">
    © 2026 Dr Merwan Roudane · Released under the
    <a href="{GH}/blob/main/LICENSE">MIT License</a>.<br>
    References: Kumar, Chaturvedi &amp; Afifa (2016); Kumar &amp; Agiwal (2019);
    Meligkotsidou, Tzavalis &amp; Vrontos; Koop, Leon-Gonzalez &amp; Strachan (2006).
  </p>
</div></footer>

</body>
</html>
"""

if __name__ == "__main__":
    out = os.path.join(HERE, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(HTML)
    # GitHub Pages: skip Jekyll so underscores / folders serve as-is
    open(os.path.join(HERE, ".nojekyll"), "w").close()
    print("wrote", os.path.relpath(out))
    print("wrote", os.path.relpath(os.path.join(HERE, ".nojekyll")))
