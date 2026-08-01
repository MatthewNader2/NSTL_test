import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

// Global error handlers for unhandled exceptions
window.onerror = (message, source, lineno, colno, error) => {
  console.error(`[NSTL Global Error] ${message} at ${source}:${lineno}:${colno}`, error);
};
window.addEventListener("unhandledrejection", (event) => {
  console.error("[NSTL Unhandled Promise Rejection]", event.reason);
});

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
