module.exports = {
  apps: [
    {
      name: 'bertnews',
      script: 'server/index.mjs',
      env: {
        PORT: 3000,
        HF_HOME: '/app/cache/hf',
        GATED_ALPHA: 0.1,
        GATED_TAU: 0.1,
        GATED_DECAY: 0.01,
        GATED_GAMMA: 2.0,
        GATED_TOPK: 0,
      },
    },
  ],
};


