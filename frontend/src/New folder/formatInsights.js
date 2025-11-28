export function formatInsights(mlInsights) {
  if (!mlInsights) return {};

  return {
    target: mlInsights.target_column,
    problemType: mlInsights.problem_type,
    bestModel: mlInsights.best_model?.name,
    bestScore: mlInsights.best_model?.score,
    featureImportances: mlInsights.feature_importances || [],
    samplePredictions: mlInsights.sample_predictions || [],
  };
}
