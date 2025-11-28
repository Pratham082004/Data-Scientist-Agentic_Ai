import React from "react";
import { Box, Typography, Paper } from "@mui/material";
import FileUploader from "../components/FileUploader/FileUploader";

export default function UploadPage() {
  return (
    <Box sx={{ p: 4 }}>
      <Paper sx={{ p: 4, borderRadius: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 700, mb: 2 }}>
          Upload Dataset
        </Typography>

        <Typography sx={{ mb: 2 }}>
          Upload a CSV file to start the analysis pipeline.
        </Typography>

        <FileUploader />
      </Paper>
    </Box>
  );
}
