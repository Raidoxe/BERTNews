import express from 'express';
import Database from 'better-sqlite3';
import crypto from 'node:crypto';
import { pipeline } from '@xenova/transformers';
import path from 'node:path';
import RSSParser from 'rss-parser';

const PORT = process.env.PORT || 3000;
const DB_PATH = process.env.DB_PATH || '/home/oliver/Documents/BERTnews/data/app.db';
// Gated learning hyperparameters (can be overridden by env)
const GATED_ALPHA = process.env.GATED_ALPHA ? Number(process.env.GATED_ALPHA) : 0.1;
const GATED_TAU = process.env.GATED_TAU ? Number(process.env.GATED_TAU) : 0.1;
const GATED_DECAY = process.env.GATED_DECAY ? Number(process.env.GATED_DECAY) : 0.01;
const GATED_GAMMA = process.env.GATED_GAMMA ? Number(process.env.GATED_GAMMA) : 2.0;
const GATED_TOPK = process.env.GATED_TOPK ? Number(process.env.GATED_TOPK) : 0; // 0 = disabled

const db = new Database(DB_PATH);

db.exec(`
CREATE TABLE IF NOT EXISTS label_cache (
  id INTEGER PRIMARY KEY,
  label_set_hash TEXT NOT NULL,
  article_index INTEGER NOT NULL,
  scores_json TEXT NOT NULL,
  UNIQUE(label_set_hash, article_index)
);
CREATE TABLE IF NOT EXISTS label_sets (
  label_set_hash TEXT PRIMARY KEY,
  labels_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profiles (
  id TEXT PRIMARY KEY,
  label_set_hash TEXT NOT NULL,
  vector_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS read_history (
  user_id TEXT NOT NULL,
  label_set_hash TEXT NOT NULL,
  article_index INTEGER NOT NULL,
  feedback TEXT NOT NULL,
  ts INTEGER NOT NULL,
  PRIMARY KEY (user_id, label_set_hash, article_index)
);
CREATE TABLE IF NOT EXISTS read_history_id (
  user_id TEXT NOT NULL,
  label_set_hash TEXT NOT NULL,
  article_id TEXT NOT NULL,
  feedback TEXT NOT NULL,
  ts INTEGER NOT NULL,
  PRIMARY KEY (user_id, label_set_hash, article_id)
);
CREATE TABLE IF NOT EXISTS article_embeddings (
  id TEXT PRIMARY KEY,
  title TEXT,
  description TEXT,
  link TEXT,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,
  updated_at INTEGER NOT NULL
);
`);

const app = express();
app.use(express.json({ limit: '2mb' }));

const inMemoryCache = new Map(); // key: labelSetHash|articleIndex -> scores
// Legacy CSV cache removed. All content comes from DB via RSS ingest.

function hashLabels(labels) {
  const norm = [...labels].map(s => s.trim().toLowerCase()).sort().join('|');
  return crypto.createHash('sha256').update(norm).digest('hex').slice(0, 16);
}

let zeroShot = null;
async function getZeroShot() {
  if (!zeroShot) {
    zeroShot = await pipeline('zero-shot-classification', 'Xenova/bart-large-mnli');
  }
  return zeroShot;
}

let embedder = null;
async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return embedder;
}

const labelEmbeddingsCache = new Map(); // key: labelSetHash -> { labels: string[], labelToVec: Map, combinedFrom: 'profile'|'uniform', combinedVec: Float32Array }

function bufferToFloat32Array(buf) {
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function l2norm(vec) {
  let s = 0; for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i]; return Math.sqrt(s);
}

function dot(a, b) {
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s;
}

function addScaled(acc, vec, scale) {
  for (let i = 0; i < acc.length; i++) acc[i] += vec[i] * scale;
}

/**
 * Apply gated, sparse update to a user preference vector.
 *
 * For each label i:
 *  - If s_i >= tau: u_i <- clip(u_i + alpha * y * (s_i ** gamma), -1, 1)
 *  - Else:          u_i <- u_i * (1 - decay)
 * Only labels present in `labels` are processed; others remain unchanged.
 *
 * @param {Record<string, number>} currentU - existing user vector (label -> weight in [-1,1])
 * @param {Record<string, number>} scores - article label scores s_i in [0,1]
 * @param {string[]} labels - label set ordering
 * @param {1|-1} y - +1 for like, -1 for dislike
 * @param {{alpha:number,tau:number,decay:number,gamma:number}} params - hyperparameters
 * @returns {Record<string, number>} new user vector
 */
function updateProfileGated(currentU, scores, labels, y, params) {
  const alpha = params.alpha ?? GATED_ALPHA;
  const tau = params.tau ?? GATED_TAU;
  const decay = params.decay ?? GATED_DECAY;
  const gamma = params.gamma ?? GATED_GAMMA;
  const out = { ...currentU };
  for (const label of labels) {
    const ui = out[label] ?? 0;
    const si = Math.max(0, Math.min(1, (scores?.[label] ?? 0)));
    let newUi;
    if (si >= tau) {
      newUi = ui + alpha * y * Math.pow(si, gamma);
    } else {
      newUi = ui * (1 - decay);
    }
    if (newUi < -1) newUi = -1; if (newUi > 1) newUi = 1;
    out[label] = newUi;
  }
  return out;
}

/**
 * Sparsify a score vector by thresholding and optional top-K pruning.
 * Keeps only labels with s_i >= tau (or |s_i| >= tau if signed), sets others to 0.
 * If topK > 0, keeps only the topK by value magnitude.
 * @param {Record<string, number>} scores
 * @param {number} tau
 * @param {number} topK
 * @returns {Record<string, number>}
 */
function sparsifyScores(scores, tau = GATED_TAU, topK = GATED_TOPK) {
  if (!scores) return {};
  const entries = Object.entries(scores)
    .map(([l, v]) => [l, Number(v) || 0])
    .filter(([, v]) => Math.abs(v) >= tau);
  if (topK > 0 && entries.length > topK) {
    entries.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    entries.length = topK;
  }
  const out = {};
  for (const [l, v] of entries) out[l] = v;
  return out;
}

function stripHtml(input) {
  if (!input) return '';
  return String(input).replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim();
}

function extractMeta(html) {
  const out = { title: '', description: '' };
  try {
    const ogTitle = html.match(/<meta[^>]*property=["']og:title["'][^>]*content=["']([^"']+)["'][^>]*>/i);
    const metaTitle = html.match(/<title[^>]*>([^<]+)<\/title>/i);
    const ogDesc = html.match(/<meta[^>]*property=["']og:description["'][^>]*content=["']([^"']+)["'][^>]*>/i);
    const metaDesc = html.match(/<meta[^>]*name=["']description["'][^>]*content=["']([^"']+)["'][^>]*>/i);
    const ldHeadline = html.match(/"headline"\s*:\s*"([^"]{5,})"/i);
    const ldDesc = html.match(/"description"\s*:\s*"([^"]{10,})"/i);
    out.title = stripHtml((ogTitle?.[1] || ldHeadline?.[1] || metaTitle?.[1] || '').slice(0, 500));
    out.description = stripHtml((ogDesc?.[1] || ldDesc?.[1] || metaDesc?.[1] || '').slice(0, 2000));
  } catch {}
  return out;
}

function deriveTitleFromURL(link) {
  try {
    const u = new URL(link);
    const parts = u.pathname.split('/').filter(Boolean);
    const last = decodeURIComponent(parts[parts.length - 1] || u.hostname);
    const cleaned = last.replace(/[-_]+/g, ' ').replace(/\.[a-z0-9]{2,4}$/i, '');
    return cleaned || u.hostname;
  } catch {
    return link;
  }
}

app.post('/topics/score_batch', async (req, res) => {
  try {
    const { articles, labels, multi_label = true, min_score = 0.05 } = req.body || {};
    if (!Array.isArray(labels) || labels.length === 0) return res.status(400).json({ error: 'labels required' });
    if (!Array.isArray(articles) || articles.length === 0) return res.status(400).json({ error: 'articles required' });

    const labelSetHash = hashLabels(labels);
    // persist label set for reuse
    db.prepare('INSERT OR IGNORE INTO label_sets (label_set_hash, labels_json) VALUES (?, ?)')
      .run(labelSetHash, JSON.stringify(labels));
    const model = await getZeroShot();

    const results = [];
    for (let i = 0; i < articles.length; i++) {
      const { index, title, description } = articles[i];
      const text = [title, description].filter(Boolean).join(' — ');
      const cacheKey = `${labelSetHash}|${index}`;

      let cached = inMemoryCache.get(cacheKey);
      if (!cached) {
        // try DB
        const row = db.prepare('SELECT scores_json FROM label_cache WHERE label_set_hash = ? AND article_index = ?').get(labelSetHash, index);
        if (row) cached = JSON.parse(row.scores_json);
      }

      if (!cached) {
        const out = await model(text, labels, { multi_label });
        const scores = {};
        for (let k = 0; k < out.labels.length; k++) {
          const label = out.labels[k];
          const score = out.scores[k];
          if (score >= min_score) scores[label] = score;
        }
        cached = scores;
        inMemoryCache.set(cacheKey, cached);
        db.prepare('INSERT OR REPLACE INTO label_cache (label_set_hash, article_index, scores_json) VALUES (?, ?, ?)')
          .run(labelSetHash, index, JSON.stringify(cached));
      }

      results.push({ index, scores: cached });
    }

    res.json({ labelSetHash, results });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Register label set (store labels and return hash)
app.post('/labels/register', (req, res) => {
  try {
    const { labels } = req.body || {};
    if (!Array.isArray(labels) || labels.length === 0) return res.status(400).json({ error: 'labels required' });
    const labelSetHash = hashLabels(labels);
    db.prepare('INSERT OR IGNORE INTO label_sets (label_set_hash, labels_json) VALUES (?, ?)')
      .run(labelSetHash, JSON.stringify(labels));
    res.json({ labelSetHash });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Serve static client
app.use(express.static(path.resolve('/home/oliver/Documents/BERTnews/web')));

// Legacy CSV endpoints removed; all articles come from DB via RSS ingest

app.post('/profiles/from_interactions', (req, res) => {
  try {
    const { user_id, labelSetHash, interactions, method = 'sum' } = req.body || {};
    if (!user_id || !labelSetHash || !Array.isArray(interactions)) return res.status(400).json({ error: 'user_id, labelSetHash, interactions required' });

    const vector = {};
    for (const it of interactions) {
      const { scores, weight = 1 } = it;
      for (const [label, val] of Object.entries(scores || {})) {
        vector[label] = (vector[label] || 0) + (method === 'mean' ? val : val * weight);
      }
    }

    db.prepare('INSERT OR REPLACE INTO profiles (id, label_set_hash, vector_json) VALUES (?, ?, ?)')
      .run(user_id, labelSetHash, JSON.stringify(vector));

    res.json({ user_id, labelSetHash, vector });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Migrate a user's profile from one label set to a new one (keep intersection, init new to 0)
app.post('/profiles/migrate', (req, res) => {
  try {
    const { user_id, fromLabelSetHash, toLabels } = req.body || {};
    if (!user_id || !Array.isArray(toLabels) || toLabels.length === 0) {
      return res.status(400).json({ error: 'user_id and toLabels required' });
    }
    const toHash = hashLabels(toLabels);

    // Ensure label set row exists for destination
    db.prepare('INSERT OR IGNORE INTO label_sets (label_set_hash, labels_json) VALUES (?, ?)')
      .run(toHash, JSON.stringify(toLabels));

    // Load old profile if provided
    let oldVec = {};
    if (fromLabelSetHash) {
      const oldRow = db.prepare('SELECT vector_json FROM profiles WHERE id = ? AND label_set_hash = ?').get(user_id, fromLabelSetHash);
      if (oldRow) oldVec = JSON.parse(oldRow.vector_json);
    }

    // Build new vector: keep intersection, new labels to 0
    const newVec = {};
    for (const l of toLabels) newVec[l] = oldVec[l] || 0;

    db.prepare('INSERT OR REPLACE INTO profiles (id, label_set_hash, vector_json) VALUES (?, ?, ?)')
      .run(user_id, toHash, JSON.stringify(newVec));

    res.json({ user_id, fromLabelSetHash, toLabelSetHash: toHash, vector: newVec });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Online feedback update: like/dislike an article to update user profile
app.post('/profiles/feedback', async (req, res) => {
  try {
    const { user_id, labelSetHash, article_id, feedback, alpha = 0.1 } = req.body || {};
    if (!user_id || !labelSetHash || !['like','dislike'].includes(feedback)) {
      return res.status(400).json({ error: 'user_id, labelSetHash, feedback required' });
    }

    // fetch label set
    const ls = db.prepare('SELECT labels_json FROM label_sets WHERE label_set_hash = ?').get(labelSetHash);
    if (!ls) return res.status(400).json({ error: 'Unknown labelSetHash' });
    const labels = JSON.parse(ls.labels_json);

    // Compute article scores/vector via DB id
    let scores = null;
    if (article_id) {
      const model = await getZeroShot();
      const row = db.prepare('SELECT title, description FROM article_embeddings WHERE id = ?').get(article_id);
      if (!row) return res.status(400).json({ error: 'Unknown article_id' });
      const text = [row.title || '', row.description || ''].filter(Boolean).join(' — ');
      const out = await model(text, labels, { multi_label: true });
      scores = {};
      for (let k = 0; k < out.labels.length; k++) scores[out.labels[k]] = out.scores[k];
      // get precomputed article vector
      const vrow = db.prepare('SELECT vector FROM article_embeddings WHERE id = ?').get(article_id);
      var articleVec = vrow ? bufferToFloat32Array(vrow.vector) : null;
    } else {
      return res.status(400).json({ error: 'article_id required' });
    }

    // Sparsify by s_i threshold/topK
    scores = sparsifyScores(scores, GATED_TAU, GATED_TOPK);

    // Further gate updates by cosine contribution (only update labels that contributed)
    if (articleVec) {
      let cached = labelEmbeddingsCache.get(labelSetHash);
      if (!cached) {
        const ef = await getEmbedder();
        const labelToVec = new Map();
        for (const l of labels) {
          const out = await ef(l, { pooling: 'mean', normalize: true });
          labelToVec.set(l, out.data);
        }
        cached = { labels, labelToVec };
        labelEmbeddingsCache.set(labelSetHash, cached);
      }
      const effScores = { ...scores };
      for (const l of labels) {
        const lv = cached.labelToVec.get(l);
        if (!lv) continue;
        const sim = dot(lv, articleVec);
        if (Math.abs(sim) < GATED_TAU) effScores[l] = 0;
      }
      scores = effScores;
    }

    // Get current user profile (or zeros)
    const pRow = db.prepare('SELECT vector_json FROM profiles WHERE id = ? AND label_set_hash = ?').get(user_id, labelSetHash);
    const u = pRow ? JSON.parse(pRow.vector_json) : {};

    // Gated sparse update
    const y = feedback === 'like' ? 1 : -1;
    const updated = updateProfileGated(u, scores, labels, y, { alpha, tau: GATED_TAU, decay: GATED_DECAY, gamma: GATED_GAMMA });

    db.prepare('INSERT OR REPLACE INTO profiles (id, label_set_hash, vector_json) VALUES (?, ?, ?)')
      .run(user_id, labelSetHash, JSON.stringify(updated));
    // Mark read by article_id
    if (article_id) {
      db.prepare('INSERT OR REPLACE INTO read_history_id (user_id, label_set_hash, article_id, feedback, ts) VALUES (?, ?, ?, ?, ?)')
        .run(user_id, labelSetHash, article_id, feedback, Date.now());
    }

    res.json({ user_id, labelSetHash, vector: updated });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

function cosineSim(vecA, vecB) {
  let dot = 0, na = 0, nb = 0;
  const labels = new Set([...Object.keys(vecA), ...Object.keys(vecB)]);
  for (const l of labels) {
    const a = vecA[l] || 0; const b = vecB[l] || 0;
    dot += a * b; na += a * a; nb += b * b;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb) || 1;
  return dot / denom;
}

app.post('/reco/rank', (req, res) => {
  try {
    const { user_id, labelSetHash, candidates, topk = 10, similarity = 'dot' } = req.body || {};
    if (!user_id || !labelSetHash || !Array.isArray(candidates)) return res.status(400).json({ error: 'user_id, labelSetHash, candidates required' });

    const row = db.prepare('SELECT vector_json FROM profiles WHERE id = ? AND label_set_hash = ?').get(user_id, labelSetHash);
    const readSet = new Set(db.prepare('SELECT article_index FROM read_history WHERE user_id = ?').all(user_id).map(r => r.article_index));
    if (!row) {
      // Cold-start fallback: rank by aggregate label score (sum), explain using top label scores
      const pool = candidates.filter(c => !readSet.has(c.index));
      const cold = pool.map(c => {
        const entries = Object.entries(c.scores || {});
        const score = entries.reduce((s, [, v]) => s + v, 0);
        const explanation = entries
          .map(([l, v]) => ({ label: l, weight: v, pref: 0 }))
          .sort((a, b) => b.weight - a.weight);
        return { index: c.index, score, explanation, cold_start: true };
      }).sort((a, b) => b.score - a.score).slice(0, topk);

      // 5% exploration: replace last with a random unseen candidate
      if (pool.length > cold.length && Math.random() < 0.05) {
        const remaining = pool.filter(p => !cold.some(x => x.index === p.index));
        if (remaining.length) {
          const rnd = remaining[Math.floor(Math.random() * remaining.length)];
          const entries = Object.entries(rnd.scores || {});
          const score = entries.reduce((s, [, v]) => s + v, 0);
          const explanation = entries.map(([l, v]) => ({ label: l, weight: v, pref: 0 })).sort((a, b) => b.weight - a.weight);
          // Place exploration item at the top to ensure visibility
          cold[0] = { index: rnd.index, score, explanation, cold_start: true, exploration: true };
        }
      }
      return res.json({ items: cold });
    }
    const userVec = JSON.parse(row.vector_json);

    const pool = candidates.filter(c => !readSet.has(c.index));
    const scored = pool.map(c => {
      // Sparsify scores for ranking as well
      const sparseScores = sparsifyScores(c.scores || {}, GATED_TAU, GATED_TOPK);
      const score = similarity === 'dot'
        ? Object.entries(sparseScores).reduce((s, [l, v]) => s + (userVec[l] || 0) * v, 0)
        : cosineSim(userVec, sparseScores);
      // Sparsify by tau for explanation/score interpretation
      const explanation = Object.entries(c.scores || {})
        .map(([l, v]) => ({ label: l, weight: (userVec[l] || 0) * (v >= GATED_TAU ? v : 0), pref: (userVec[l] || 0) }))
        .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
      return { index: c.index, score, explanation };
    }).sort((a, b) => b.score - a.score).slice(0, topk);

    // 5% exploration: replace last with a random unseen candidate
    if (pool.length > scored.length && Math.random() < 0.05) {
      const remaining = pool.filter(p => !scored.some(x => x.index === p.index));
      if (remaining.length) {
        const rnd = remaining[Math.floor(Math.random() * remaining.length)];
        const score = similarity === 'dot'
          ? Object.entries(rnd.scores || {}).reduce((s, [l, v]) => s + (userVec[l] || 0) * v, 0)
          : cosineSim(userVec, rnd.scores || {});
        const sparseScoresRnd = sparsifyScores(rnd.scores || {}, GATED_TAU, GATED_TOPK);
        const explanation = Object.entries(rnd.scores || {})
          .map(([l, v]) => ({ label: l, weight: (userVec[l] || 0) * (sparseScoresRnd[l] ?? 0), pref: (userVec[l] || 0) }))
          .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
        // Place exploration item at the top to ensure visibility
        scored[0] = { index: rnd.index, score, explanation, exploration: true };
      }
    }

    res.json({ items: scored });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Ranking over embeddings: build a user embedding from label embeddings weighted by user profile, then dot with article embeddings (entire dataset)
app.post('/reco/rank_embeddings', async (req, res) => {
  try {
    const { user_id, labelSetHash, topk = 10 } = req.body || {};
    if (!user_id || !labelSetHash) return res.status(400).json({ error: 'user_id and labelSetHash required' });

    // labels
    const ls = db.prepare('SELECT labels_json FROM label_sets WHERE label_set_hash = ?').get(labelSetHash);
    if (!ls) return res.status(400).json({ error: 'Unknown labelSetHash' });
    const labels = JSON.parse(ls.labels_json);

    // user profile (may be empty)
    const pRow = db.prepare('SELECT vector_json FROM profiles WHERE id = ? AND label_set_hash = ?').get(user_id, labelSetHash);
    const u = pRow ? JSON.parse(pRow.vector_json) : {};
    const hasProfile = !!pRow;

    // label embeddings (cached)
    let cached = labelEmbeddingsCache.get(labelSetHash);
    if (!cached) {
      const ef = await getEmbedder();
      const labelToVec = new Map();
      for (const l of labels) {
        const out = await ef(l, { pooling: 'mean', normalize: true });
        labelToVec.set(l, out.data);
      }
      cached = { labels, labelToVec };
      labelEmbeddingsCache.set(labelSetHash, cached);
    }

    // Build user embedding as weighted sum of label embeddings; cold-start = uniform weights 1.0
    const sampleVec = cached.labelToVec.get(labels[0]);
    const userVec = new Float32Array(sampleVec.length);
    let anyWeight = false;
    for (const l of labels) {
      const w = hasProfile ? (u[l] ?? 0) : 1.0; // strict neutrality after profile exists; uniform only at cold-start
      if (w !== 0) { addScaled(userVec, cached.labelToVec.get(l), w); anyWeight = true; }
    }
    const nrm = l2norm(userVec);
    if (nrm > 0) { for (let i = 0; i < userVec.length; i++) userVec[i] /= nrm; }

    // Exclude read by article_id
    const readIdSet = new Set(db.prepare('SELECT article_id FROM read_history_id WHERE user_id = ?').all(user_id).map(r => r.article_id));

    // Score entire dataset using stored embeddings
    const rows = db.prepare('SELECT id, title, description, link, dim, vector FROM article_embeddings').all();
    const scored = [];
    for (const r of rows) {
      if (readIdSet.has(r.id)) continue;
      const vec = bufferToFloat32Array(r.vector);
      const score = dot(userVec, vec);
      const explanation = labels.map(l => {
        const lv = cached.labelToVec.get(l);
        const sim = dot(lv, vec);
        const gatedSim = (Math.abs(sim) >= GATED_TAU) ? sim : 0;
        const pref = hasProfile ? (u[l] ?? 0) : 1.0;
        const contrib = pref * gatedSim;
        return { label: l, weight: contrib, sim, pref };
      }).sort((a,b) => Math.abs(b.weight) - Math.abs(a.weight));
      scored.push({ id: r.id, title: r.title, description: r.description, link: r.link, score, explanation });
    }

    scored.sort((a,b) => b.score - a.score);
    let top = scored.slice(0, topk);
    // 5% exploration: place a random unseen candidate at the top
    if (scored.length > top.length && Math.random() < 0.05) {
      const remaining = scored.slice(top.length);
      const rnd = remaining[Math.floor(Math.random() * remaining.length)];
      top = [{ ...rnd, exploration: true }, ...top.slice(0, Math.max(0, topk - 1))];
    }
    res.json({ items: top });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Scan popular RSS feeds, dedupe by link, embed and store new articles
app.post('/ingest/rss_scan', async (req, res) => {
  try {
    const parser = new RSSParser();
    const feeds = (
      req.body?.feeds || [
        // General/world
        'https://www.reuters.com/world/rss',
        'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
        'https://feeds.washingtonpost.com/rss/world',
        'https://www.aljazeera.com/xml/rss/all.xml',
        'https://www.theguardian.com/world/rss',
        'https://feeds.a.dj.com/rss/RSSWorldNews.xml',
        'https://feeds.skynews.com/feeds/rss/world.xml',
        'https://www.cnn.com/rss/edition_world.rss',
        'https://feeds.bbci.co.uk/news/world/rss.xml',
        'https://feeds.npr.org/1001/rss.xml',
        'https://www.cbsnews.com/latest/rss/main',
        'https://www.usatoday.com/rss/news/',
        'https://www.telegraph.co.uk/news/rss.xml',
        'https://www.independent.co.uk/news/world/rss',
        'https://time.com/feed/',
        'https://www.politico.com/rss/politics-news.xml',
        'https://www.theatlantic.com/feed/all/',
        // Tech/business (helps variety)
        'https://feeds.feedburner.com/TechCrunch/',
        'https://www.theverge.com/rss/index.xml',
        'https://www.cnbc.com/id/100003114/device/rss/rss.html'
      ]
    ).slice(0, 40);

    const existingRows = db.prepare('SELECT id, link, title, description FROM article_embeddings').all();
    const known = new Map(existingRows.map(r => [r.link, r]));
    const ef = await getEmbedder();
    const upsert = db.prepare('INSERT OR REPLACE INTO article_embeddings (id, title, description, link, dim, vector, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)');
    const updateMeta = db.prepare('UPDATE article_embeddings SET title = ?, description = ?, updated_at = ? WHERE id = ?');

    let fetched = 0, inserted = 0, skipped = 0;
    for (const url of feeds) {
      let feed;
      try { feed = await parser.parseURL(url); } catch { continue; }
      for (const it of feed.items || []) {
        let link = it.link || '';
        if (!link && Array.isArray(it.links) && it.links[0]?.url) link = it.links[0].url;
        if (!link && it.guid && /^https?:\/\//i.test(it.guid)) link = it.guid;
        if (!link) { skipped++; continue; }

        const rawTitle = it.title || it['dc:title'] || '';
        const rawDesc = it.contentSnippet || it.summary || it.content || it['content:encodedSnippet'] || it['content:encoded'] || '';
        const title = stripHtml(rawTitle);
        const desc = stripHtml(rawDesc);

        const existing = known.get(link);

        // If metadata is missing from RSS, try fetching page immediately
        let finalTitle = title;
        let finalDesc = desc;
        if (!finalTitle || !finalDesc) {
          try {
            const controller = new AbortController();
            const t = setTimeout(() => controller.abort(), 8000);
            const resp = await fetch(link, { signal: controller.signal });
            clearTimeout(t);
            if (resp.ok) {
              const html = await resp.text();
              const meta = extractMeta(html);
              if (!finalTitle && meta.title) finalTitle = meta.title;
              if (!finalDesc && meta.description) finalDesc = meta.description;
            }
          } catch {}
        }

        if (existing) {
          // Backfill missing metadata if we now have better fields (possibly fetched)
          const needTitle = (!existing.title || existing.title.trim() === '') && finalTitle;
          const needDesc = (!existing.description || existing.description.trim() === '') && finalDesc;
          if (needTitle || needDesc) {
            updateMeta.run(needTitle ? finalTitle : existing.title, needDesc ? finalDesc : existing.description, Date.now(), existing.id);
            existing.title = needTitle ? finalTitle : existing.title;
            existing.description = needDesc ? finalDesc : existing.description;
          }
          skipped++; continue;
        }

        if (!finalTitle) finalTitle = deriveTitleFromURL(link);
        const text = [finalTitle, finalDesc].filter(Boolean).join(' — ');
        const out = await ef(text || (title || ''), { pooling: 'mean', normalize: true });
        const vec = out.data;
        const buf = Buffer.from(new Float32Array(vec).buffer);
        const id = link; // use link as id
        upsert.run(id, finalTitle || title || '', finalDesc || desc || '', link, vec.length, buf, Date.now());
        known.set(link, { id, link, title: finalTitle || title || '', description: finalDesc || desc || '' });
        fetched++; inserted++;
      }
    }
    res.json({ feeds: feeds.length, fetched, inserted, skipped });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Repair existing DB rows lacking title/description by fetching page metadata
app.post('/admin/repair_empty_articles', async (req, res) => {
  try {
    const rows = db.prepare("SELECT id, link FROM article_embeddings WHERE (title IS NULL OR title = '') OR (description IS NULL OR description = '')").all();
    if (rows.length === 0) return res.json({ updated: 0 });
    let updated = 0;
    const upd = db.prepare('UPDATE article_embeddings SET title = ?, description = ?, updated_at = ? WHERE id = ?');
    for (const r of rows) {
      try {
        const controller = new AbortController();
        const t = setTimeout(() => controller.abort(), 8000);
        const resp = await fetch(r.link, { signal: controller.signal });
        clearTimeout(t);
        if (!resp.ok) continue;
        const html = await resp.text();
        const meta = extractMeta(html);
        if (meta.title || meta.description) {
          upd.run(meta.title, meta.description, Date.now(), r.id);
          updated++;
        }
      } catch {}
    }
    res.json({ updated });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Fetch read list for a user and label set
app.get('/read/list', (req, res) => {
  try {
    const { user_id } = req.query || {};
    if (!user_id) return res.status(400).json({ error: 'user_id required' });

    // From full dataset (id-based)
    const idRows = db.prepare('SELECT article_id, feedback, ts FROM read_history_id WHERE user_id = ? ORDER BY ts DESC')
      .all(user_id);
    const idItems = idRows.map(r => {
      const art = db.prepare('SELECT id, title, description, link FROM article_embeddings WHERE id = ?').get(r.article_id);
      if (!art) return null;
      return { id: art.id, title: art.title, description: art.description, link: art.link, feedback: r.feedback, ts: r.ts };
    }).filter(Boolean);

    // Merge and dedupe by id
    const seen = new Set();
    const merged = [];
    for (const it of idItems) {
      if (seen.has(it.id)) continue;
      seen.add(it.id);
      merged.push(it);
    }

    // Sort by time desc
    merged.sort((a,b) => (b.ts||0) - (a.ts||0));

    res.json({ items: merged });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});
