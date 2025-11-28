import React from "react";
import { Box, CircularProgress, Typography } from "@mui/material";

export default function Loader({ text = "Processing..." }) {
  return (
    <Box
      sx={{
        mt: 5,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 2,
      }}
    >
      <CircularProgress size={50} />
      <Typography variant="body1">{text}</Typography>
    </Box>
  );
}
