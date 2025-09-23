import fs from 'node:fs';
import path from 'node:path';
import { parse } from 'csv-parse';
import { pipeline } from '@xenova/transformers';

const CSV_PATH = path.resolve('/home/oliver/Documents/BERTnews/bbc_news.csv');
const OUTPUT_PATH = path.resolve('/home/oliver/Documents/BERTnews/out/topics_titles_first100.jsonl');
const MAX_ROWS = 100;
const K_NEIGHBORS = 10;

async function computeEmbeddings(texts) {
  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  const embeddings = [];
  for (const t of texts) {
    const input = t || '';
    const res = await embedder(input, { pooling: 'mean', normalize: true });
    embeddings.push(res.data); // Float32Array
  }
  return embeddings;
}

function cosineSimilarity(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // normalized vectors → dot = cosine
}

function topKNeighborsForAll(embeddings, k) {
  const n = embeddings.length;
  const neighbors = new Array(n);
  for (let i = 0; i < n; i++) {
    const sims = [];
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      sims.push([j, cosineSimilarity(embeddings[i], embeddings[j])]);
    }
    sims.sort((a, b) => b[1] - a[1]);
    neighbors[i] = sims.slice(0, k);
  }
  return neighbors;
}

async function main() {
  fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });

  const input = fs.createReadStream(CSV_PATH);
  const parser = input.pipe(parse({ columns: true, skip_empty_lines: true }));

  const rows = [];
  for await (const row of parser) {
    if (rows.length >= MAX_ROWS) break;
    rows.push({ title: (row.title || '').toString(), index: rows.length });
  }

  const titles = rows.map(r => r.title);
  const embeddings = await computeEmbeddings(titles);
  const neighbors = topKNeighborsForAll(embeddings, K_NEIGHBORS);

  const out = fs.createWriteStream(OUTPUT_PATH, { flags: 'w' });
  for (let i = 0; i < rows.length; i++) {
    const neigh = neighbors[i].map(([j, sim]) => ({ index: j, title: rows[j].title, cosine: sim }));
    out.write(JSON.stringify({ index: i, title: rows[i].title, neighbors: neigh }) + '\n');
  }
  out.end();
  console.log(`Computed neighbors for ${rows.length} titles. Output: ${OUTPUT_PATH}`);
}

main().catch((err) => {
  console.error('topics script failed:', err);
  process.exit(1);
});


