import React, { useEffect, useState } from "react";
import { Box, Typography, CircularProgress } from "@mui/material";
import mlApi from "../api/mlApi";
import BarChart from "../components/Chart/BarChart";
import ModelScoreTable from "../components/ModelScoreTable/ModelScoreTable";

export default function MLReport() {
  const [ml, setML] = useState(null);

  useEffect(() => {
    mlApi.getInsights().then((res) => setML(res));
  }, []);

  if (!ml)
    return (
      <Box sx={{ textAlign: "center", mt: 4 }}>
        <CircularProgress />
      </Box>
    );

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
        Machine Learning Report
      </Typography>

      <ModelScoreTable scores={ml.cv_scores} best={ml.best_model} />

      {ml.feature_importances && (
        <BarChart
          title="Feature Importances"
          data={ml.feature_importances}
          xKey="feature"
          yKey="importance"
        />
      )}
    </Box>
  );
}
