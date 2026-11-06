import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "C:/Users/wjdrm/Downloads/throughput_후처리_업데이트.xlsx";
const resultsPath = "C:/Users/wjdrm/.codex/attachments/195b37e2-fb3e-470f-aa03-27c7a4e5639d/pasted-text.txt";
const outputDir = "C:/Users/wjdrm/bp_bench_pkg/outputs/019fa153-f4b8-7e50-ac60-f5ba837f894c";
const outputPath = `${outputDir}/throughput_후처리_업데이트_최신결과.xlsx`;
const previewPath = `${outputDir}/throughput_후처리_업데이트_최신결과.png`;
const focusedPreviewPath = `${outputDir}/throughput_후처리_업데이트_최신결과_표검증.png`;

function parseRate(text) {
  const match = text.trim().match(/^([0-9.]+)\s*(K|M)?\/s$/);
  if (!match) throw new Error(`Unrecognized rate: ${text}`);
  const scale = match[2] === "M" ? 1_000_000 : match[2] === "K" ? 1_000 : 1;
  return Number(match[1]) * scale;
}

function formatScientific(value) {
  return value.toExponential(4).replace(/e([+-])(\d+)$/, (_, sign, digits) => `e${sign}${digits.padStart(2, "0")}`);
}

function wilson95(probability, n = 100_000_000) {
  const z = 1.96;
  const k = Math.round(probability * n);
  const denominator = n + z * z;
  const center = (k + z * z / 2) / denominator;
  const half = z * Math.sqrt(k * (n - k) / n + z * z / 4) / denominator;
  return `[${formatScientific(center - half)}, ${formatScientific(center + half)}]`;
}

const resultText = await fs.readFile(resultsPath, "utf8");
const resultMap = new Map();

for (const line of resultText.split(/\r?\n/)) {
  const parts = line.split("|").map((part) => part.trim());
  if (parts.length !== 12 || !/^0\.00[13579]$/.test(parts[0])) continue;

  const p = parts[0];
  const iteration = Number(parts[1]);
  const lerProbability = Number(parts[11]) / 100;
  resultMap.set(`${p}|${iteration}`, {
    prior: Number(parts[2]),
    cycles: Number(parts[3]),
    cpu1: parseRate(parts[6]),
    cpu16: parseRate(parts[7]),
    dpu1: parseRate(parts[8]),
    dpu2560: parseRate(parts[9]),
    ratio: Number(parts[10].replace(/x$/i, "")),
    ler: lerProbability,
    speedUs: Number(parts[5]),
    ci: wilson95(lerProbability),
  });
}

if (resultMap.size !== 50) {
  throw new Error(`Expected 50 final result rows, found ${resultMap.size}`);
}

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItem("throughput_후처리");

const originalHeaders = sheet.getRange("A1:L1").values[0];
const keys = sheet.getRange("A2:C51").values;
const updatedValues = keys.map(([pValue, iterationValue, priorValue], index) => {
  const p = Number(pValue).toFixed(3);
  const iteration = Number(iterationValue);
  const result = resultMap.get(`${p}|${iteration}`);
  if (!result) throw new Error(`No result for row ${index + 2}: p=${p}, iter=${iteration}`);
  if (Number(priorValue) !== result.prior) {
    throw new Error(`Prior mismatch at row ${index + 2}: workbook=${priorValue}, result=${result.prior}`);
  }
  return [
    result.cycles,
    result.cpu1,
    result.cpu16,
    result.dpu1,
    result.dpu2560,
    result.ratio,
    result.ler,
    result.speedUs,
    result.ci,
  ];
});

sheet.getRange("D2:L51").values = updatedValues;

const finalHeaders = sheet.getRange("A1:L1").values[0];
if (JSON.stringify(originalHeaders) !== JSON.stringify(finalHeaders)) {
  throw new Error("Header values changed unexpectedly");
}

const keyCheck = await workbook.inspect({
  kind: "table",
  range: "throughput_후처리!A1:L51",
  include: "values,formulas",
  tableMaxRows: 51,
  tableMaxCols: 12,
  maxChars: 26000,
});
console.log("=== UPDATED_TABLE ===");
console.log(keyCheck.ndjson);

const styleCheck = await workbook.inspect({
  kind: "computedStyle",
  sheetId: "throughput_후처리",
  range: "A1:L3",
  maxChars: 8000,
});
console.log("=== STYLE_CHECK ===");
console.log(styleCheck.ndjson);

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log("=== FORMULA_ERRORS ===");
console.log(formulaErrors.ndjson);

const formulaCheck = await workbook.inspect({
  kind: "formula",
  sheetId: "throughput_후처리",
  range: "A1:T80",
  maxChars: 3000,
  options: { maxResults: 50 },
});
console.log("=== FORMULA_CHECK ===");
console.log(formulaCheck.ndjson);

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({
  sheetName: "throughput_후처리",
  autoCrop: "all",
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const exportedInput = await FileBlob.load(outputPath);
const exportedWorkbook = await SpreadsheetFile.importXlsx(exportedInput);
const exportedCheck = await exportedWorkbook.inspect({
  kind: "table,formula,computedStyle,drawing",
  sheetId: "throughput_후처리",
  range: "A1:T80",
  maxChars: 14000,
  tableMaxRows: 12,
  tableMaxCols: 20,
  options: { maxResults: 100 },
});
console.log("=== EXPORTED_REIMPORT_CHECK ===");
console.log(exportedCheck.ndjson);

const focusedPreview = await exportedWorkbook.render({
  sheetName: "throughput_후처리",
  range: "A1:L15",
  scale: 2,
  format: "png",
});
await fs.writeFile(focusedPreviewPath, new Uint8Array(await focusedPreview.arrayBuffer()));

console.log("=== OUTPUT ===");
console.log(outputPath);
console.log(previewPath);
console.log(focusedPreviewPath);
