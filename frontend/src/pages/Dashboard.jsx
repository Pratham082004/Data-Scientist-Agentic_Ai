import React, { useEffect, useState } from "react";
import { Box, Typography, Grid, CircularProgress } from "@mui/material";
import edaApi from "../api/edaApi";
import mlApi from "../api/mlApi";
import BarChart from "../components/Chart/BarChart";
import PieChart from "../components/Chart/PieChart";
import ModelScoreTable from "../components/ModelScoreTable/ModelScoreTable";

export default function Dashboard() {
  const [eda, setEda] = useState(null);
  const [ml, setML] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([edaApi.getInsights(), mlApi.getInsights()]).then(
      ([edaRes, mlRes]) => {
        setEda(edaRes);
        setML(mlRes);
        setLoading(false);
      }
    );
  }, []);

  if (loading)
    return (
      <Box sx={{ textAlign: "center", mt: 4 }}>
        <CircularProgress />
      </Box>
    );

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 4 }}>
        Analytics Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* EDA PIE CHART */}
        {eda?.distribution && (
          <Grid item xs={12} md={6}>
            <PieChart
              data={eda.distribution}
              labelKey="label"
              valueKey="value"
              title="Class Distribution"
            />
          </Grid>
        )}

        {/* ML FEATURE IMPORTANCE */}
        {ml?.feature_importances && (
          <Grid item xs={12} md={6}>
            <BarChart
              title="Feature Importances"
              data={ml.feature_importances}
              xKey="feature"
              yKey="importance"
            />
          </Grid>
        )}

        {/* MODEL SCORES */}
        <Grid item xs={12}>
          <ModelScoreTable scores={ml.cv_scores} best={ml.best_model} />
        </Grid>
      </Grid>
    </Box>
  );
}
