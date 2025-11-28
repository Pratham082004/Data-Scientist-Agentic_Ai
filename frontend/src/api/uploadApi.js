import api from "./client";

export const uploadDataset = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  return api.post("/upload", formData);
};
