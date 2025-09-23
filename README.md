# BERTnews

Minimal, interpretable, label-driven news recommendations (Node + @xenova/transformers).

## Run locally

```
npm run server   # dev server (nodemon)
```
App: http://localhost:3000 (UI in /web/)

## Env

Copy env.example → .env (or use PM2 ecosystem config):
- PORT=3000
- HF_HOME=/app/cache/hf
- GATED_ALPHA=0.1
- GATED_TAU=0.1
- GATED_DECAY=0.01
- GATED_GAMMA=2.0
- GATED_TOPK=0

## Deploy on one EC2 (MVP)

1) Ubuntu 22.04 (t3.small). Open 80/443. Point DNS to EC2.
2) Install Node, pm2, nginx, certbot.
3) Clone & start:
```
cd /opt/bertnews/app
npm ci
pm2 start ecosystem.config.cjs
pm2 save && pm2 startup
```
4) Nginx reverse proxy:
- Copy deploy/nginx.conf.example → /etc/nginx/sites-available/bertnews (edit domain)
- Enable & reload: ln -s, nginx -t, reload
- TLS: certbot --nginx -d yourdomain -d www.yourdomain

## Persistence
- HF cache: /opt/bertnews/app/cache/hf (set HF_HOME)
- SQLite DB: /opt/bertnews/app/data/app.db (optional S3 backup via cron)

## Docker (optional)
```
docker build -t bertnews .
docker run -p 3000:3000 -e HF_HOME=/app/cache/hf \
  -v $(pwd)/cache/hf:/app/cache/hf -v $(pwd)/data:/app/data bertnews
```

Notes: gated learning (tau=0.1, gamma=2.0) and embedding-based ranking are default. Explanations show contribution, cosine, weight.
