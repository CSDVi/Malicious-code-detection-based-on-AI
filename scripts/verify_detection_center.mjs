import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("C:/Users/35132/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/.pnpm/playwright-core@1.61.1/node_modules/playwright-core");

const root = "D:/Z_TEMP_photo/AI/code/artifacts";
const browser = await chromium.launch({
    headless: true,
    executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });

await page.goto("http://127.0.0.1:3000/login", { waitUntil: "networkidle" });
await page.fill('input[name="username"]', "admin");
await page.fill('input[name="password"]', "admin123");
await Promise.all([
    page.waitForLoadState("networkidle"),
    page.click('button[type="submit"]'),
]);

await page.goto("http://127.0.0.1:3000/attack/", { waitUntil: "networkidle" });
await page.evaluate(() => {
    const transfer = new DataTransfer();
    transfer.items.add(new File(["safe = 1\nos.system(user_input)\nprint(safe)\n"], "command.py", { type: "text/x-python" }));
    document.querySelector(".compact-upload").dispatchEvent(new DragEvent("drop", {
        bubbles: true, dataTransfer: transfer,
    }));
});
if (!(await page.locator(".compact-upload [data-file-label]").innerText()).includes("command.py")) {
    throw new Error("Single-file drop did not populate the upload input");
}
await page.locator('input[name="mode"][value="quick"]').evaluate((input) => {
    input.checked = true;
    input.dispatchEvent(new Event("change", { bubbles: true }));
});
await Promise.all([
    page.waitForLoadState("networkidle"),
    page.click('button[type="submit"]'),
]);

await page.locator(".evidence-item").first().waitFor();
const resultText = await page.locator("main").innerText();
if (!resultText.includes("第 2 行") || !resultText.includes("可疑度 100/100")) {
    throw new Error("Evidence line or suspicion score was not rendered");
}
if (!resultText.includes("系统命令") || !resultText.includes("使用固定命令和参数白名单")) {
    throw new Error("Chinese explanation or repair advice was not rendered");
}
if (resultText.includes("行为类别、CWE、ATT&CK")) {
    throw new Error("Removed behavior/CWE/ATT&CK panel is still visible");
}
await page.screenshot({ path: path.join(root, "detection-result-desktop.png"), fullPage: true });

await page.goto("http://127.0.0.1:3000/attack/history", { waitUntil: "networkidle" });
await page.locator('a[title="查看报告"]').first().click();
await page.locator(".evidence-item").first().waitFor();
const historyDetailText = await page.locator("main").innerText();
if (!historyDetailText.includes("第 2 行") || historyDetailText.includes("行为类别、CWE、ATT&CK")) {
    throw new Error("Historical report did not use the simplified evidence layout");
}

await page.goto("http://127.0.0.1:3000/attack/project", { waitUntil: "networkidle" });
const projectText = await page.locator("main").innerText();
if (!projectText.includes("拖入 ZIP 项目包或点击选择")) {
    throw new Error("Project drop zone is missing");
}
await page.evaluate(() => {
    const transfer = new DataTransfer();
    transfer.items.add(new File([new Uint8Array([80, 75, 3, 4])], "sample-project.zip", { type: "application/zip" }));
    document.querySelector(".project-upload").dispatchEvent(new DragEvent("drop", {
        bubbles: true, dataTransfer: transfer,
    }));
});
if (!(await page.locator(".project-upload [data-file-label]").innerText()).includes("sample-project.zip")) {
    throw new Error("Project drop did not populate the upload input");
}
await page.setViewportSize({ width: 390, height: 844 });
await page.reload({ waitUntil: "networkidle" });
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
if (overflow > 1) throw new Error(`Mobile layout overflows by ${overflow}px`);
await page.screenshot({ path: path.join(root, "project-upload-mobile.png"), fullPage: true });

await browser.close();
console.log(JSON.stringify({ result: "ok", screenshots: [
    path.join(root, "detection-result-desktop.png"),
    path.join(root, "project-upload-mobile.png"),
] }));
