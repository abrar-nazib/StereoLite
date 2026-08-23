async (page) => {
  // Edit NAME before each call. Renders NAME.excalidraw, POSTs the SVG to the local sink (port 8766)
  // as NAME.svg, and also saves a 1x screenshot NAME_1x.png for a quick look.
  const NAME = "architecture";
  const DIAGRAM = "/paper/figures/excalidraw/" + NAME + ".excalidraw";
  const OUT = "/home/abrar/Research/stero_research_claude/paper/figures/excalidraw/" + NAME + "_1x.png";
  await page.setViewportSize({ width: 1600, height: 800 });
  await page.goto("http://127.0.0.1:8765/paper/figures/excalidraw/render_template_local.html?t=" + Date.now());
  await page.waitForFunction("window.__moduleReady === true", null, { timeout: 90000 });
  const merr = await page.evaluate("window.__moduleError || null");
  if (merr) return "MODULE FAILED: " + merr;
  const res = await page.evaluate(async (path) => {
    const r = await fetch(path + "?t=" + Date.now());
    const data = await r.json();
    return await window.renderDiagram(data);
  }, DIAGRAM);
  if (!res || !res.success) return "RENDER FAILED: " + JSON.stringify(res);
  await page.waitForFunction("window.__renderComplete === true", null, { timeout: 30000 });
  const saved = await page.evaluate(async (name) => {
    const svg = document.querySelector("#root svg").outerHTML;
    const r = await fetch("http://127.0.0.1:8766/save?name=" + name + ".svg", { method: "POST", body: svg });
    return await r.text();
  }, NAME);
  const svg = page.locator("#root svg");
  const box = await svg.boundingBox();
  await svg.screenshot({ path: OUT });
  return "OK " + saved + " box=" + JSON.stringify(box) + " dpr=" + (await page.evaluate("window.devicePixelRatio"));
}
