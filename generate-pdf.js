// Save this as generate-pdf.js in a Node.js environment

const puppeteer = require("puppeteer");
const fs = require("fs");

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function generatePDF(outputPath, reportUrl) {
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox"]
  });
  const page = await browser.newPage();
  await page.setUserAgent(
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
  );
  await page.setViewport({ width: 1280, height: 800 });
  await page.goto(reportUrl, { waitUntil: "networkidle2" });
  await sleep(2000);
  await page.pdf({
    path: outputPath,
    format: "A4",
    printBackground: true,
    margin: { top: "20mm", bottom: "20mm", left: "15mm", right: "15mm" },
    pageRanges: "1-8"
  });
  await browser.close();
  console.log(`PDF saved to ${outputPath}`);
}

const outputPath = process.argv[2];
const reportUrl = process.argv[3];
if (!outputPath || !reportUrl) {
  console.error("Usage: node generate-pdf.js <output-path> <report-url>");
  process.exit(1);
}
generatePDF(outputPath, reportUrl).catch(err => {
  console.error("Error generating PDF:", err);
  process.exit(1);
});
