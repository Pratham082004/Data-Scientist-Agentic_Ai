import React from "react";
import { Card, CardContent, Typography, Box } from "@mui/material";

export default function ReportCard({ title, value, subtitle, children }) {
  return (
    <Card
      elevation={3}
      sx={{
        minWidth: 250,
        borderRadius: 3,
        p: 1,
      }}
    >
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          {title}
        </Typography>

        {value && (
          <Typography
            variant="h4"
            sx={{ mt: 1, fontWeight: "bold", color: "primary.main" }}
          >
            {value}
          </Typography>
        )}

        {subtitle && (
          <Typography variant="subtitle2" sx={{ opacity: 0.7, mt: 0.5 }}>
            {subtitle}
          </Typography>
        )}

        <Box sx={{ mt: 2 }}>{children}</Box>
      </CardContent>
    </Card>
  );
}
