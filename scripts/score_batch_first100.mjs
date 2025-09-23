import fs from 'node:fs';
import path from 'node:path';

const INPUT_JSONL = path.resolve('/home/oliver/Documents/BERTnews/out/ner_merged_first100.jsonl');
const OUTPUT_JSONL = path.resolve('/home/oliver/Documents/BERTnews/out/zero_shot_scores_first100.jsonl');
const API_URL = process.env.API_URL || 'http://localhost:3000/topics/score_batch';

function parseCliLabels(argv) {
  // Accept: node script.mjs --labels Economy,Politics,War or --label Economy --label Politics
  const labels = [];
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--labels' || arg === '-l') {
      const val = argv[++i] || '';
      val.split(',').forEach(s => s && labels.push(s.trim()));
    } else if (arg === '--label') {
      const val = argv[++i] || '';
      if (val) labels.push(val.trim());
    }
  }
  return labels;
}

async function main() {
  const content = fs.readFileSync(INPUT_JSONL, 'utf8');
  const rows = content.split('\n').filter(Boolean).map(l => JSON.parse(l));

  let labels = parseCliLabels(process.argv);
  if (labels.length === 0) {
    labels = (process.env.LABELS || 'Economy,Politics,Climate,Tech,Sport,Health,War,Energy,Education,Crime')
      .split(',').map(s => s.trim()).filter(Boolean);
  }
  if (labels.length === 0) throw new Error('No labels provided');

  const body = {
    labels,
    multi_label: true,
    min_score: Number(process.env.MIN_SCORE || 0.05),
    articles: rows.map(r => ({ index: r.index, title: r.title, description: r.description }))
  };

  const resp = await fetch(API_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`API error ${resp.status}: ${txt}`);
  }
  const data = await resp.json();

  fs.mkdirSync(path.dirname(OUTPUT_JSONL), { recursive: true });
  const out = fs.createWriteStream(OUTPUT_JSONL, { flags: 'w' });
  for (const r of data.results) {
    out.write(JSON.stringify({ index: r.index, scores: r.scores, labelSetHash: data.labelSetHash, labels }) + '\n');
  }
  out.end();
  console.log(`Wrote scores for ${data.results.length} articles to ${OUTPUT_JSONL}`);
}

main().catch(e => {
  console.error('batch scoring failed:', e);
  process.exit(1);
});
