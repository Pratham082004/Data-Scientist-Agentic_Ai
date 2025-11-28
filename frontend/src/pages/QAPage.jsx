import React from "react";
import { Box, Typography, Paper } from "@mui/material";
import QnAChat from "../components/QnAChat/QnAChat";

export default function QAPage() {
  return (
    <Box sx={{ p: 4 }}>
      <Paper sx={{ p: 4, borderRadius: 3, mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 2 }}>
          Ask Questions About Your Data
        </Typography>
        <Typography>
          This chat uses your EDA + ML insight JSON reports to answer questions.
        </Typography>
      </Paper>

      <QnAChat />
    </Box>
  );
}
