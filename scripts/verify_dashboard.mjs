import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("C:/Users/35132/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/.pnpm/playwright-core@1.61.1/node_modules/playwright-core");

const artifactRoot = "D:/Z_TEMP_photo/AI/code/artifacts";
const browser = await chromium.launch({
    headless: true,
    executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
});
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });

await page.goto("http://127.0.0.1:3000/login", { waitUntil: "domcontentloaded" });
await page.fill('input[name="username"]', "admin");
await page.fill('input[name="password"]', "admin123");
await Promise.all([
    page.waitForLoadState("domcontentloaded"),
    page.click('button[type="submit"]'),
]);
await page.goto("http://127.0.0.1:3000/", { waitUntil: "domcontentloaded" });

const mainText = await page.locator("main.landing-page").innerText();
for (const removed of ["RECENT SIGNALS", "本地证据引擎可用", "检测任务", "高风险记录", "平均风险分", "5 个模型已就绪"]) {
    if (mainText.includes(removed)) throw new Error(`Removed dashboard copy is still visible: ${removed}`);
}
for (const expected of ["最近检测记录", "4 个模型已就绪", "查看模型运行状态"]) {
    if (!mainText.includes(expected)) throw new Error(`Expected dashboard copy is missing: ${expected}`);
}
if (await page.locator(".signal-strip").count()) throw new Error("Removed dashboard metric strip still exists");

const panelBox = await page.locator(".recent-panel").boundingBox();
const statusBox = await page.locator(".recent-model-status").boundingBox();
const readyBox = await page.locator(".recent-model-ready").boundingBox();
const linkBox = await page.locator(".recent-model-status a").boundingBox();
if (!panelBox || !statusBox || !readyBox || !linkBox) throw new Error("Dashboard status layout cannot be measured");
if (statusBox.y < panelBox.y + panelBox.height) throw new Error("Model status row is not below the recent-record panel");
if (readyBox.x >= linkBox.x) throw new Error("Model readiness and status link are not arranged left-to-right");

await page.screenshot({ path: path.join(artifactRoot, "dashboard-desktop.png"), fullPage: true });

await page.goto("http://127.0.0.1:3000/attack/models", { waitUntil: "domcontentloaded" });
if (await page.locator(".model-runtime-grid .runtime-item").count() !== 4) {
    throw new Error("Model center does not show exactly four model families");
}
if ((await page.locator("main.main-content").innerText()).includes("规则引擎")) {
    throw new Error("Rule engine is still presented as a model");
}

await page.goto("http://127.0.0.1:3000/", { waitUntil: "domcontentloaded" });
await page.setViewportSize({ width: 390, height: 844 });
await page.reload({ waitUntil: "domcontentloaded" });
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
if (overflow > 1) throw new Error(`Mobile dashboard overflows by ${overflow}px`);
await page.screenshot({ path: path.join(artifactRoot, "dashboard-mobile.png"), fullPage: true });

await browser.close();
console.log(JSON.stringify({
    result: "ok",
    screenshots: [
        path.join(artifactRoot, "dashboard-desktop.png"),
        path.join(artifactRoot, "dashboard-mobile.png"),
    ],
}));
