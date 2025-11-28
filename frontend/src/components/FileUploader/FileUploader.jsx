import React, { useState } from "react";
import { Box, Button, Typography, Paper } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";

export default function FileUploader({ onUpload }) {
  const [file, setFile] = useState(null);

  const handleChoose = (e) => {
    const selected = e.target.files[0];
    if (selected && selected.name.endsWith(".csv")) {
      setFile(selected);
    } else {
      alert("Only CSV files are allowed!");
    }
  };

  const handleUpload = () => {
    if (file) onUpload(file);
  };

  return (
    <Paper
      elevation={3}
      sx={{
        p: 4,
        maxWidth: 500,
        mx: "auto",
        mt: 4,
        textAlign: "center",
      }}
    >
      <Typography variant="h6" sx={{ mb: 2 }}>
        Upload Dataset (CSV)
      </Typography>

      <Button
        variant="contained"
        component="label"
        startIcon={<CloudUploadIcon />}
        sx={{ mb: 2 }}
      >
        Choose File
        <input hidden type="file" accept=".csv" onChange={handleChoose} />
      </Button>

      {file && (
        <Typography variant="body1" sx={{ mb: 2 }}>
          Selected: <strong>{file.name}</strong>
        </Typography>
      )}

      <Button
        variant="contained"
        color="success"
        disabled={!file}
        onClick={handleUpload}
      >
        Upload & Process
      </Button>
    </Paper>
  );
}
