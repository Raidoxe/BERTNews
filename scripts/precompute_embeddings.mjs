import fs from 'node:fs';
import path from 'node:path';
import Database from 'better-sqlite3';
import { parse as parseCSV } from 'csv-parse/sync';
import { pipeline } from '@xenova/transformers';

const CSV_PATH = path.resolve('/home/oliver/Documents/BERTnews/bbc_news.csv');
const DB_PATH = process.env.DB_PATH || '/home/oliver/Documents/BERTnews/data/app.db';
const MAX_ROWS = Number(process.env.MAX_ROWS || 0); // 0 = all

function articleIdFromRow(row) {
  // Prefer stable guid/link if present, else fallback to row index during this run
  return (row.guid || row.link || row.title).toString();
}

function float32ArrayToBuffer(arr) {
  return Buffer.from(new Float32Array(arr).buffer);
}

async function main() {
  const db = new Database(DB_PATH);
  db.exec('PRAGMA journal_mode = WAL;');
  const ensure = db.prepare(`CREATE TABLE IF NOT EXISTS article_embeddings (
    id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    link TEXT,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    updated_at INTEGER NOT NULL
  );`);
  ensure.run();

  const csv = fs.readFileSync(CSV_PATH, 'utf8');
  const rows = parseCSV(csv, { columns: true, skip_empty_lines: true });
  const slice = MAX_ROWS > 0 ? rows.slice(0, MAX_ROWS) : rows;

  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  const selectStmt = db.prepare('SELECT id FROM article_embeddings WHERE id = ?');
  const upsertStmt = db.prepare('INSERT OR REPLACE INTO article_embeddings (id, title, description, link, dim, vector, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)');

  let processed = 0, skipped = 0;
  for (let i = 0; i < slice.length; i++) {
    const r = slice[i];
    const id = articleIdFromRow(r);
    if (selectStmt.get(id)) { skipped++; continue; }

    const text = [r.title || '', r.description || ''].filter(Boolean).join(' — ');
    const res = await embedder(text, { pooling: 'mean', normalize: true });
    const vec = res.data; // Float32Array
    const buf = float32ArrayToBuffer(vec);
    upsertStmt.run(id, r.title || '', r.description || '', r.link || '', vec.length, buf, Date.now());
    processed++;
    if ((processed + skipped) % 200 === 0) {
      console.log(`Processed: ${processed}, skipped: ${skipped}`);
    }
  }

  console.log(`Done. New embeddings: ${processed}, existing skipped: ${skipped}`);
}

main().catch(err => {
  console.error('precompute embeddings failed:', err);
  process.exit(1);
});


