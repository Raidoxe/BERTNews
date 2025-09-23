FROM node:20-slim
WORKDIR /app

# Install deps
COPY package*.json ./
RUN npm ci --omit=dev

# App
COPY . .
RUN mkdir -p /app/data /app/cache/hf

ENV PORT=3000
ENV HF_HOME=/app/cache/hf

EXPOSE 3000
CMD ["node","server/index.mjs"]


