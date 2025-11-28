import api from "./client";

export const askQuestion = (question) =>
  api.post("/qa", { question });
