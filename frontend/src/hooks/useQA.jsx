import { useState } from "react";
import qaApi from "../api/qaApi";

export default function useQA() {
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);

  const askQuestion = async (question) => {
    setLoading(true);

    // Push user message locally
    setMessages((prev) => [...prev, { from: "user", text: question }]);

    try {
      const res = await qaApi.ask(question);

      setMessages((prev) => [
        ...prev,
        { from: "ai", text: res.answer || "(no answer)" },
      ]);

      return res;
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { from: "ai", text: "Error: Could not answer question." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return { askQuestion, loading, messages };
}
