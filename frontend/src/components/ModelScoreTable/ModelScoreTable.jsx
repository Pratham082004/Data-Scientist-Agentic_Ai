import React from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Box,
  Typography,
} from "@mui/material";

export default function ModelScoreTable({ scores = {}, bestModel = "" }) {
  const modelEntries = Object.entries(scores);

  if (modelEntries.length === 0) {
    return (
      <Box sx={{ mt: 3, textAlign: "center" }}>
        <Typography variant="body1" color="text.secondary">
          No model scores available.
        </Typography>
      </Box>
    );
  }

  return (
    <TableContainer component={Paper} elevation={3} sx={{ mt: 3, borderRadius: 3 }}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: 700 }}>Model Name</TableCell>
            <TableCell sx={{ fontWeight: 700 }}>Score</TableCell>
            <TableCell sx={{ fontWeight: 700 }}>Status</TableCell>
          </TableRow>
        </TableHead>

        <TableBody>
          {modelEntries.map(([name, score]) => {
            const isBest = name === bestModel;

            return (
              <TableRow
                key={name}
                sx={{
                  backgroundColor: isBest ? "rgba(25, 118, 210, 0.08)" : "inherit",
                }}
              >
                <TableCell sx={{ fontSize: "1rem", fontWeight: isBest ? 600 : 400 }}>
                  {name}
                </TableCell>

                <TableCell sx={{ fontSize: "1rem" }}>
                  {typeof score === "number" ? score.toFixed(4) : score}
                </TableCell>

                <TableCell>
                  {isBest ? (
                    <Chip label="Best Model" color="primary" size="small" />
                  ) : (
                    <Chip label="Evaluated" variant="outlined" size="small" />
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
