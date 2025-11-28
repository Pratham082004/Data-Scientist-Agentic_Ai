const API_BASE = "http://127.0.0.1:5000";

const mlApi = {
    runML: async (fileData) => {
        const response = await fetch(`${API_BASE}/ml`, {
            method: "POST",
            body: fileData,
        });
        return response.json();
    },

    getMLReport: async () => {
        const response = await fetch(`${API_BASE}/ml/report`);
        return response.json();
    }
};

export default mlApi;
