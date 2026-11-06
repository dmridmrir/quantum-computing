import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "C:/Users/wjdrm/Downloads/throughput_후처리_업데이트.xlsx";
const previewDir = "C:/Users/wjdrm/bp_bench_pkg/.artifact_spreadsheet_work/previews_before";

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const overview = await workbook.inspect({
  kind: "workbook,sheet,table,drawing",
  maxChars: 12000,
  tableMaxRows: 12,
  tableMaxCols: 14,
  tableMaxCellChars: 120,
});
console.log("=== OVERVIEW ===");
console.log(overview.ndjson);

const sheets = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 6000 });
console.log("=== SHEETS ===");
console.log(sheets.ndjson);

const valuesAndFormulas = await workbook.inspect({
  kind: "region,formula,computedStyle",
  sheetId: "throughput_후처리",
  range: "A1:T80",
  maxChars: 16000,
  tableMaxRows: 15,
  tableMaxCols: 20,
  options: { maxResults: 200 },
});
console.log("=== REGION_FORMULAS_STYLES ===");
console.log(valuesAndFormulas.ndjson);

try {
  const chart = workbook.resolve("ch/43q5kf2x");
  console.log("=== CHART ===");
  console.log(JSON.stringify({
    type: chart.type,
    title: chart.title,
    categories: chart.categories,
    series: chart.series?.items?.map((s) => ({
      name: s.name,
      formula: s.formula,
      categoryFormula: s.categoryFormula,
    })) ?? [],
  }, null, 2));
} catch (error) {
  console.log("=== CHART ERROR ===");
  console.log(String(error));
}

await fs.mkdir(previewDir, { recursive: true });
const sheetNames = [];
for (const line of sheets.ndjson.split(/\r?\n/)) {
  if (!line.trim()) continue;
  try {
    const record = JSON.parse(line);
    const name = record.name ?? record.sheetName ?? record.title;
    if (typeof name === "string" && !sheetNames.includes(name)) sheetNames.push(name);
  } catch {}
}

for (let i = 0; i < sheetNames.length; i++) {
  const name = sheetNames[i];
  const preview = await workbook.render({ sheetName: name, autoCrop: "all", scale: 1, format: "png" });
  const safe = name.replace(/[\\/:*?"<>|]/g, "_");
  await fs.writeFile(`${previewDir}/${String(i + 1).padStart(2, "0")}_${safe}.png`, new Uint8Array(await preview.arrayBuffer()));
}

console.log("=== PREVIEWS ===");
console.log(sheetNames.map((name, i) => `${i + 1}: ${name}`).join("\n"));
