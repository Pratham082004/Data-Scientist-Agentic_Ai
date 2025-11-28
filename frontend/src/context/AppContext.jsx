import React, { createContext, useContext, useState } from "react";

const AppContext = createContext();

export function AppProvider({ children }) {
  const [file, setFile] = useState(null);
  const [edaReport, setEdaReport] = useState(null);
  const [mlReport, setMlReport] = useState(null);

  const value = {
    file,
    setFile,

    edaReport,
    setEdaReport,

    mlReport,
    setMlReport,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useAppContext() {
  return useContext(AppContext);
}
