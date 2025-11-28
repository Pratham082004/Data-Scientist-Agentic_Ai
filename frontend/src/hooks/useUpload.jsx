import { useState } from "react";
import uploadApi from "../api/uploadApi";

export default function useUpload() {
  const [uploading, setUploading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const uploadFile = async (file) => {
    setUploading(true);
    setError(null);

    try {
      const res = await uploadApi.uploadDataset(file);
      setResponse(res);
      return res;
    } catch (err) {
      setError(err);
      throw err;
    } finally {
      setUploading(false);
    }
  };

  return { uploadFile, uploading, response, error };
}
