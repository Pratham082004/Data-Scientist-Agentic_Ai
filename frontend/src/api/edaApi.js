const API_BASE = "http://127.0.0.1:5000";

const edaApi = {
    runEDA: async (fileData) => {
        const response = await fetch(`${API_BASE}/eda`, {
            method: "POST",
            body: fileData,
        });
        return response.json();
    },

    getEDAReport: async () => {
        const response = await fetch(`${API_BASE}/eda/report`);
        return response.json();
    }
};

export default edaApi;
