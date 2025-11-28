import React, { useState } from "react";
import {
  Paper,
  TextField,
  IconButton,
  Typography,
  Box,
  CircularProgress,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import client from "../../api/client";

export default function QnAChat() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!question.trim()) return;

    const newMsg = { sender: "user", text: question };
    setMessages((prev) => [...prev, newMsg]);
    setLoading(true);

    try {
      const response = await client.post("/qa/ask", {
        question: question,
      });

      const aiMsg = {
        sender: "ai",
        text: response.data.answer,
      };

      setMessages((prev) => [...prev, aiMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { sender: "ai", text: "Error fetching answer." },
      ]);
    }

    setLoading(false);
    setQuestion("");
  };

  return (
    <Paper sx={{ p: 3, borderRadius: 3 }}>
      <Typography variant="h5" sx={{ mb: 2, fontWeight: 700 }}>
        Ask Any Question About Your Data 📊
      </Typography>

      {/* Message list */}
      <Box
        sx={{
          height: 350,
          overflowY: "auto",
          borderRadius: 2,
          p: 2,
          background: "#f5f5f5",
          mb: 2,
        }}
      >
        {messages.map((msg, i) => (
          <Box
            key={i}
            sx={{
              mb: 1.5,
              textAlign: msg.sender === "user" ? "right" : "left",
            }}
          >
            <Paper
              elevation={1}
              sx={{
                display: "inline-block",
                p: 1.5,
                borderRadius: 2,
                background:
                  msg.sender === "user" ? "#1976d2" : "white",
                color: msg.sender === "user" ? "#fff" : "#333",
              }}
            >
              {msg.text}
            </Paper>
          </Box>
        ))}

        {loading && (
          <Box sx={{ textAlign: "center", mt: 2 }}>
            <CircularProgress size={24} color="primary" />
          </Box>
        )}
      </Box>

      {/* Input box */}
      <Box sx={{ display: "flex", gap: 1 }}>
        <TextField
          fullWidth
          placeholder="Ask something about EDA or ML insights..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <IconButton color="primary" onClick={sendMessage}>
          <SendIcon />
        </IconButton>
      </Box>
    </Paper>
  );
}
