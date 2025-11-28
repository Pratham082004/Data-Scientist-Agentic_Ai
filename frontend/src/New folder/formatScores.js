export function formatScores(scores) {
  if (!scores) return [];

  return Object.entries(scores).map(([model, score]) => ({
    model,
    score: Number(score.toFixed(4))
  }));
}
