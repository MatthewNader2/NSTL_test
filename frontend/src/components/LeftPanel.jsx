import { useState, useCallback } from "react";
import { useStore } from "../store";
import { runPrompt } from "../hooks/useApi";
import {
  Send,
  ChevronLeft,
  ChevronRight,
  X,
  Cpu,
  RefreshCw,
} from "lucide-react";

export default function LeftPanel({ open, onToggle }) {
  const [prompt, setPrompt] = useState("");
  const addLog = useStore((s) => s.addLog);
  const setActivePath = useStore((s) => s.setActivePath);
  const setVirtualEdges = useStore((s) => s.setVirtualEdges);
  const setGeneratedCode = useStore((s) => s.setGeneratedCode);
  const [history, setHistory] = useState([]);

  const handleRun = useCallback(async () => {
    if (!prompt.trim()) return;
    setHistory((prev) => [...prev, prompt]);
    addLog({
      type: "system",
      msg: `Prompt: "${prompt}"`,
      time: new Date().toLocaleTimeString(),
    });
    try {
      const data = await runPrompt(prompt);
      for (const log of data.logs) {
        addLog({
          type: log.type,
          msg: log.msg,
          time: new Date().toLocaleTimeString(),
        });
      }
      setActivePath(data.path);
      setVirtualEdges(new Set(data.virtual_edges));
      setGeneratedCode(data.code);
    } catch (err) {
      addLog({
        type: "warn",
        msg: `[API ERROR] ${err.message}`,
        time: new Date().toLocaleTimeString(),
      });
    }
    setPrompt("");
  }, [prompt]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleRun();
    }
  };

  return (
    <div
      className={`left-panel glass-panel ${open ? "" : "collapsed"}`}
      style={{ borderRight: open ? "1px solid var(--glass-border)" : "none" }}
    >
      {open && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            height: "100%",
            padding: "0.5rem",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "0.5rem",
            }}
          >
            <span
              style={{
                fontSize: "0.625rem",
                color: "var(--text-secondary)",
                letterSpacing: 1,
              }}
            >
              CONSOLE
            </span>
            <button
              onClick={onToggle}
              style={{ color: "var(--text-secondary)" }}
            >
              <X size={14} />
            </button>
          </div>

          {/* History */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              marginBottom: "0.5rem",
              fontSize: "0.6875rem",
              color: "var(--text-secondary)",
            }}
          >
            {history.map((h, i) => (
              <div
                key={i}
                style={{
                  marginBottom: "0.25rem",
                  padding: "0.125rem 0.25rem",
                  background: "rgba(0,255,255,0.05)",
                  borderRadius: 4,
                }}
              >
                {h}
              </div>
            ))}
          </div>

          {/* Quick chips */}
          <div
            style={{
              display: "flex",
              gap: "0.25rem",
              flexWrap: "wrap",
              marginBottom: "0.375rem",
            }}
          >
            {[
              "load csv and clean nulls",
              "full ETL pipeline",
              "ML pipeline",
            ].map((ex) => (
              <button
                key={ex}
                onClick={() => setPrompt(ex)}
                style={{
                  fontSize: "0.5625rem",
                  padding: "0.125rem 0.375rem",
                  background: "rgba(0,229,255,0.08)",
                  borderRadius: 4,
                  color: "var(--text-secondary)",
                }}
              >
                {ex}
              </button>
            ))}
          </div>

          {/* Prompt input */}
          <div style={{ display: "flex", gap: "0.25rem", alignItems: "flex-end" }}>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe your pipeline..."
              rows={2}
              style={{
                flex: 1,
                resize: "none",
                background: "rgba(0,0,0,0.3)",
                border: "1px solid var(--glass-border)",
                borderRadius: 6,
                padding: "0.375rem",
                fontSize: "0.75rem",
                color: "var(--text-primary)",
                outline: "none",
              }}
            />
            <button
              onClick={handleRun}
              disabled={!prompt.trim()}
              style={{
                padding: "0.375rem 0.625rem",
                background: "var(--accent)",
                borderRadius: 6,
                color: "#000",
                fontWeight: 700,
                opacity: prompt.trim() ? 1 : 0.4,
              }}
            >
              <Send size={14} />
            </button>
          </div>
        </div>
      )}
      {/* Toggle button when collapsed */}
      {!open && (
        <button
          onClick={onToggle}
          style={{
            position: "absolute",
            top: "0.5rem",
            left: "0.5rem",
            zIndex: 10,
            color: "var(--text-secondary)",
          }}
        >
          <ChevronRight size={16} />
        </button>
      )}
    </div>
  );
}
