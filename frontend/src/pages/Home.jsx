import React from "react";
import { Box, Typography, Button, Paper } from "@mui/material";
import { Link } from "react-router-dom";

export default function Home() {
  return (
    <Box sx={{ p: 4 }}>
      <Paper sx={{ p: 4, borderRadius: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Data Scientist Agentic AI
        </Typography>

        <Typography sx={{ mt: 2, mb: 4 }}>
          Upload a dataset and let the agent perform cleaning, EDA, ML, and Q&A powered insights.
        </Typography>

        <Button variant="contained" size="large" component={Link} to="/upload">
          Get Started
        </Button>
      </Paper>
    </Box>
  );
}
