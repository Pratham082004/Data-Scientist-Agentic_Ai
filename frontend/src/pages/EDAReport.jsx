import React, { useEffect, useState } from "react";
import { Box, Typography, CircularProgress } from "@mui/material";
import edaApi from "../api/edaApi";
import BarChart from "../components/Chart/BarChart";
import PieChart from "../components/Chart/PieChart";

export default function EDAReport() {
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    edaApi.getInsights().then((res) => {
      setInsights(res);
      setLoading(false);
    });
  }, []);

  if (loading)
    return (
      <Box sx={{ textAlign: "center", mt: 4 }}>
        <CircularProgress />
      </Box>
    );

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 2 }}>
        EDA Report
      </Typography>

      {/* Example: column distribution (pie chart) */}
      {insights?.distribution && (
        <PieChart
          data={insights.distribution}
          labelKey="label"
          valueKey="value"
          title="Class Distribution"
        />
      )}

      {/* Example: numeric summary top 5 */}
      {insights?.top_variance && (
        <BarChart
          data={insights.top_variance}
          xKey="feature"
          yKey="variance"
          title="Top Feature Variance"
        />
      )}
    </Box>
  );
}
