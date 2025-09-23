import fs from 'node:fs';
import path from 'node:path';
import { parse } from 'csv-parse';
import { pipeline } from '@xenova/transformers';

const CSV_PATH = path.resolve('/home/oliver/Documents/BERTnews/bbc_news.csv');
const OUTPUT_PATH = path.resolve('/home/oliver/Documents/BERTnews/out/ner_merged_first100.jsonl');
const MAX_ROWS = 100;

function normalizeEntity(entity) {
  return entity.trim();
}

function mergeEntities(titleSpans, descSpans) {
  const mergedByNorm = new Map();

  const addSpan = (span, source) => {
    const norm = normalizeEntity(span.text || span.word || '').toString();
    if (!norm) return;
    const key = norm.toLowerCase();
    const payload = {
      text: span.text || span.word || norm,
      type: span.type || span.entity_group || span.entity || 'ENTITY',
      score: typeof span.score === 'number' ? span.score : undefined,
      source,
    };
    if (!mergedByNorm.has(key)) {
      mergedByNorm.set(key, payload);
    } else {
      const existing = mergedByNorm.get(key);
      // Prefer the one with higher score and keep type if more specific
      if ((payload.score ?? 0) > (existing.score ?? 0)) {
        mergedByNorm.set(key, { ...existing, ...payload });
      } else if (existing.type === 'MISC' && payload.type !== 'MISC') {
        mergedByNorm.set(key, { ...existing, type: payload.type });
      }
    }
  };

  for (const s of titleSpans) addSpan(s, 'title');
  for (const s of descSpans) addSpan(s, 'description');

  return Array.from(mergedByNorm.values()).sort((a, b) => a.type.localeCompare(b.type) || a.text.localeCompare(b.text));
}

// Collapse subword tokens (e.g., "Z", "##ele", "##nsky") into full words ("Zelensky")
// using BIO-style labels from `entity` (e.g., B-PER, I-PER). If labels are missing,
// treat each token independently.
function collapseSubwordEntities(spans) {
  if (!Array.isArray(spans) || spans.length === 0) return [];

  const result = [];
  let current = null;

  const appendPiece = (acc, pieceText, pieceScore) => {
    if (pieceText.startsWith('##')) {
      acc.text += pieceText.slice(2);
    } else if (acc.text.length === 0) {
      acc.text = pieceText;
    } else {
      acc.text += ' ' + pieceText;
    }
    acc.scores.push(typeof pieceScore === 'number' ? pieceScore : 0);
  };

  for (const s of spans) {
    const word = (s.word ?? s.text ?? '').toString();
    const label = (s.entity ?? s.entity_group ?? '').toString();
    const score = s.score;

    const isBIO = /^([BI])-(\w+)/.exec(label);
    if (!isBIO) {
      // Flush current if any
      if (current) {
        current.score = current.scores.length
          ? current.scores.reduce((a, b) => a + b, 0) / current.scores.length
          : undefined;
        delete current.scores;
        result.push(current);
        current = null;
      }
      // Standalone token
      if (word) {
        result.push({ text: word.replace(/^##/, ''), type: label || 'ENTITY', score });
      }
      continue;
    }

    const [, bi, type] = isBIO;
    if (bi === 'B' || !current || current.type !== type) {
      // Start new chunk
      if (current) {
        current.score = current.scores.length
          ? current.scores.reduce((a, b) => a + b, 0) / current.scores.length
          : undefined;
        delete current.scores;
        result.push(current);
      }
      current = { text: '', type, scores: [] };
      appendPiece(current, word, score);
    } else {
      // Continue current chunk
      appendPiece(current, word, score);
    }
  }

  if (current) {
    current.score = current.scores.length
      ? current.scores.reduce((a, b) => a + b, 0) / current.scores.length
      : undefined;
    delete current.scores;
    result.push(current);
  }

  return result;
}

async function main() {
  // Ensure output directory
  fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });

  // Prepare NER pipeline (downloads model on first run, cached afterwards)
  const ner = await pipeline('token-classification', 'Xenova/bert-base-NER', { aggregation_strategy: 'simple' });

  const input = fs.createReadStream(CSV_PATH);
  const parser = input.pipe(parse({ columns: true, skip_empty_lines: true }));

  const outStream = fs.createWriteStream(OUTPUT_PATH, { flags: 'w' });
  let processed = 0;

  for await (const row of parser) {
    if (processed >= MAX_ROWS) break;

    const title = (row.title || '').toString();
    const description = (row.description || '').toString();

    // Run NER on both fields
    const [titleRaw, descRaw] = await Promise.all([
      title ? ner(title) : Promise.resolve([]),
      description ? ner(description) : Promise.resolve([]),
    ]);

    // Collapse subword pieces to word-level entities
    const titleEntities = collapseSubwordEntities(titleRaw);
    const descEntities = collapseSubwordEntities(descRaw);

    // Merge
    const merged = mergeEntities(titleEntities, descEntities);

    const record = {
      index: processed,
      title,
      description,
      entities_title: titleEntities.map(e => ({ text: e.text, type: e.type, score: e.score })),
      entities_description: descEntities.map(e => ({ text: e.text, type: e.type, score: e.score })),
      entities_merged: merged,
    };

    outStream.write(JSON.stringify(record) + '\n');
    processed += 1;
  }

  outStream.end();
  console.log(`Processed ${processed} rows. Output: ${OUTPUT_PATH}`);
}

main().catch((err) => {
  console.error('NER script failed:', err);
  process.exit(1);
});


