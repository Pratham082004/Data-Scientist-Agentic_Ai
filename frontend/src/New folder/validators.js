export function isCSV(file) {
  return file && file.type === "text/csv";
}

export function isValidQuestion(text) {
  return text && text.trim().length > 3;
}

export function hasEDAReport(report) {
  return report && typeof report === "object";
}

export function hasMLReport(report) {
  return report && typeof report === "object";
}
