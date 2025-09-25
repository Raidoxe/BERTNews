import fs from 'node:fs';
import path from 'node:path';
import { parse } from 'csv-parse';
import { pipeline, AutoTokenizer } from '@xenova/transformers';

const CSV_PATH = process.env.CSV_PATH || path.resolve(process.cwd(), 'bbc_news.csv');
const OUTPUT_PATH = process.env.OUTPUT_PATH || path.resolve(process.cwd(), 'out/topic_keywords_first100.jsonl');
const MAX_ROWS = 100;
const TOP_K = 10;

// Build candidate tokens for a title: unigrams and bigrams filtered for stopwords/non-alpha
const STOPWORDS = new Set([
  'the','a','an','and','or','but','if','to','of','in','on','for','with','as','at','by','from','into','about','over','after','before','between','through','during','under','above','up','down','out','off','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','can','will','just','don','should','now','is','are','was','were','be','been','being','it','its','this','that','these','those','you','your','i','we','they','he','she','him','her','them','our','their'
]);

function tokenizeText(text) {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s\-']/gu, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

function generateCandidates(title) {
  const tokens = tokenizeText(title);
  const words = tokens.filter(t => !STOPWORDS.has(t) && /[\p{L}\p{N}]/u.test(t));
  const uniq = Array.from(new Set(words));
  const bigrams = [];
  for (let i = 0; i < tokens.length - 1; i++) {
    const a = tokens[i], b = tokens[i+1];
    if (STOPWORDS.has(a) || STOPWORDS.has(b)) continue;
    if (!/[\p{L}\p{N}]/u.test(a) || !/[\p{L}\p{N}]/u.test(b)) continue;
    bigrams.push(`${a} ${b}`);
  }
  const all = Array.from(new Set([...uniq, ...bigrams]));
  return all.slice(0, 100); // cap per title for speed
}

async function computeEmbeddings(embedder, texts) {
  const out = [];
  for (const t of texts) {
    const res = await embedder(t || '', { pooling: 'mean', normalize: true });
    out.push(res.data);
  }
  return out;
}

function cosine(a, b) {
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s;
}

function mmr(docVec, candVecs, lambda = 0.7, topK = 10) {
  const selected = [];
  const candidates = candVecs.map((v, idx) => ({ idx, sim: cosine(docVec, v) }));
  const chosen = new Set();
  while (selected.length < Math.min(topK, candidates.length)) {
    let bestIdx = -1; let bestScore = -Infinity;
    for (const { idx, sim } of candidates) {
      if (chosen.has(idx)) continue;
      let diversity = 0;
      for (const sel of selected) diversity = Math.max(diversity, cosine(candVecs[idx], candVecs[sel.idx]));
      const mmrScore = lambda * sim - (1 - lambda) * diversity;
      if (mmrScore > bestScore) { bestScore = mmrScore; bestIdx = idx; }
    }
    if (bestIdx === -1) break;
    chosen.add(bestIdx);
    selected.push({ idx: bestIdx, sim: candidates.find(c => c.idx === bestIdx).sim });
  }
  return selected;
}

async function main() {
  fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });

  const input = fs.createReadStream(CSV_PATH);
  const parser = input.pipe(parse({ columns: true, skip_empty_lines: true }));

  const rows = [];
  for await (const row of parser) {
    if (rows.length >= MAX_ROWS) break;
    rows.push({ index: rows.length, title: (row.title || '').toString() });
  }

  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  const out = fs.createWriteStream(OUTPUT_PATH, { flags: 'w' });
  for (const r of rows) {
    const candidates = generateCandidates(r.title);
    const [docVecArr] = await computeEmbeddings(embedder, [r.title]);
    const candVecs = await computeEmbeddings(embedder, candidates);
    const picks = mmr(docVecArr, candVecs, 0.7, TOP_K);
    const keywords = picks.map(p => ({ keyword: candidates[p.idx], score: p.sim }));
    out.write(JSON.stringify({ index: r.index, title: r.title, keywords }) + '\n');
  }

  out.end();
  console.log(`Extracted topic keywords for ${rows.length} titles. Output: ${OUTPUT_PATH}`);
}

main().catch(err => {
  console.error('topic keywords failed:', err);
  process.exit(1);
});


