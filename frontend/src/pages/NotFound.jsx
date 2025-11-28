import React from "react";
import { Box, Typography, Button } from "@mui/material";
import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <Box sx={{ p: 4, textAlign: "center" }}>
      <Typography variant="h3" sx={{ fontWeight: 700 }}>
        404
      </Typography>

      <Typography sx={{ mb: 2 }}>Page Not Found</Typography>

      <Button variant="contained" component={Link} to="/">
        Go Home
      </Button>
    </Box>
  );
}
