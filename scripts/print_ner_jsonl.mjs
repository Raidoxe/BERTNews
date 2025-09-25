import fs from 'node:fs';
import path from 'node:path';

const INPUT_PATH = process.argv[2]
  ? path.resolve(process.argv[2])
  : (process.env.NER_JSONL || path.resolve(process.cwd(), 'out/ner_merged_first100.jsonl'));

const TOPICS_PATH = process.argv[3]
  ? path.resolve(process.argv[3])
  : (process.env.TOPICS_TITLES_JSONL || path.resolve(process.cwd(), 'out/topics_titles_first100.jsonl'));

const KEYWORDS_PATH = process.argv[4]
  ? path.resolve(process.argv[4])
  : (process.env.TOPIC_KEYWORDS_JSONL || path.resolve(process.cwd(), 'out/topic_keywords_first100.jsonl'));

const ZSCORES_PATH = process.argv[5]
  ? path.resolve(process.argv[5])
  : (process.env.ZERO_SHOT_SCORES_JSONL || path.resolve(process.cwd(), 'out/zero_shot_scores_first100.jsonl'));

function formatEntityList(list) {
  return list
    .map((e) => {
      const text = e.text ?? e.word ?? e.entity ?? '';
      const type = e.type ?? e.entity_group ?? '';
      const score = typeof e.score === 'number' ? e.score.toFixed(3) : undefined;
      return score ? `${text} (${type}, ${score})` : `${text} (${type})`;
    })
    .join(', ');
}

function formatNeighbors(neighbors) {
  if (!Array.isArray(neighbors) || neighbors.length === 0) return '';
  return neighbors
    .slice(0, 10)
    .map((n) => `${n.index}: ${n.title} [cos=${n.cosine.toFixed(3)}]`)
    .join('\n  ');
}

function printRecord(r, topicsByIndex) {
  console.log('---');
  console.log(`Title: ${r.title}`);
  console.log(`Desc: ${r.description}`);
  console.log(`entities title: ${formatEntityList(r.entities_title ?? [])}`);
  console.log(`entities description: ${formatEntityList(r.entities_description ?? [])}`);
  console.log(`entities merged: ${formatEntityList(r.entities_merged ?? [])}`);
  const t = topicsByIndex.get(r.index);
  if (t) {
    const formatted = formatNeighbors(t.neighbors || []);
    console.log('topic neighbors:');
    if (formatted) console.log('  ' + formatted);
  }
  console.log('----');
}

function loadTopicsMap(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n').map((l) => l.trim()).filter(Boolean);
    const map = new Map();
    for (const line of lines) {
      try {
        const obj = JSON.parse(line);
        map.set(obj.index, obj);
      } catch {}
    }
    return map;
  } catch {
    return new Map();
  }
}

function main() {
  const topicsByIndex = loadTopicsMap(TOPICS_PATH);
  const keywordsByIndex = loadTopicsMap(KEYWORDS_PATH);
  const zscoresByIndex = loadTopicsMap(ZSCORES_PATH);
  const stream = fs.createReadStream(INPUT_PATH, { encoding: 'utf8' });
  let buffer = '';
  stream.on('data', (chunk) => {
    buffer += chunk;
    let idx;
    while ((idx = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      try {
        const obj = JSON.parse(line);
        // attach keywords if present
        const keyObj = keywordsByIndex.get(obj.index);
        const zs = zscoresByIndex.get(obj.index);
        if (keyObj && Array.isArray(keyObj.keywords)) {
          console.log('---');
          console.log(`Title: ${obj.title}`);
          console.log(`Desc: ${obj.description}`);
          console.log(`entities title: ${formatEntityList(obj.entities_title ?? [])}`);
          console.log(`entities description: ${formatEntityList(obj.entities_description ?? [])}`);
          console.log(`entities merged: ${formatEntityList(obj.entities_merged ?? [])}`);
          const t = topicsByIndex.get(obj.index);
          if (t) {
            const formatted = formatNeighbors(t.neighbors || []);
            console.log('topic neighbors:');
            if (formatted) console.log('  ' + formatted);
          }
          const kw = keyObj.keywords
            .slice(0, 10)
            .map(k => `${k.keyword} [sim=${(typeof k.score === 'number' ? k.score.toFixed(3) : '')}]`)
            .join(', ');
          console.log(`topic keywords: ${kw}`);
          if (zs && zs.scores) {
            const pairs = Object.entries(zs.scores).sort((a,b)=>b[1]-a[1]).slice(0,10)
              .map(([l, s]) => `${l}=${s.toFixed(3)}`).join(', ');
            console.log(`label scores: ${pairs}`);
          }
          console.log('----');
        } else {
          printRecord(obj, topicsByIndex);
        }
      } catch (err) {
        console.error('Failed to parse line:', line.slice(0, 120));
      }
    }
  });
  stream.on('end', () => {
    if (buffer.trim()) {
      try {
        const obj = JSON.parse(buffer.trim());
        const keyObj = keywordsByIndex.get(obj.index);
        const zs = zscoresByIndex.get(obj.index);
        if (keyObj && Array.isArray(keyObj.keywords)) {
          console.log('---');
          console.log(`Title: ${obj.title}`);
          console.log(`Desc: ${obj.description}`);
          console.log(`entities title: ${formatEntityList(obj.entities_title ?? [])}`);
          console.log(`entities description: ${formatEntityList(obj.entities_description ?? [])}`);
          console.log(`entities merged: ${formatEntityList(obj.entities_merged ?? [])}`);
          const t = topicsByIndex.get(obj.index);
          if (t) {
            const formatted = formatNeighbors(t.neighbors || []);
            console.log('topic neighbors:');
            if (formatted) console.log('  ' + formatted);
          }
          const kw = keyObj.keywords
            .slice(0, 10)
            .map(k => `${k.keyword} [sim=${(typeof k.score === 'number' ? k.score.toFixed(3) : '')}]`)
            .join(', ');
          console.log(`topic keywords: ${kw}`);
          if (zs && zs.scores) {
            const pairs = Object.entries(zs.scores).sort((a,b)=>b[1]-a[1]).slice(0,10)
              .map(([l, s]) => `${l}=${s.toFixed(3)}`).join(', ');
            console.log(`label scores: ${pairs}`);
          }
          console.log('----');
        } else {
          printRecord(obj, topicsByIndex);
        }
      } catch {
        // ignore trailing partial
      }
    }
  });
  stream.on('error', (err) => {
    console.error('Error reading file:', err.message);
    process.exit(1);
  });
}

main();


