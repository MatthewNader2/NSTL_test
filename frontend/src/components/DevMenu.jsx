import { useStore } from "../store";
import { X, Trash2, Copy, Check, Terminal, Network } from "lucide-react";
import { useState, useMemo, useEffect } from "react";
import { getApiBase, setApiBase } from "../hooks/useApi";

export default function DevMenu({ embedderDevice, setEmbedderDevice, llmDevice, setLlmDevice, onHardwareApply }) {
  const devMenuOpen = useStore((s) => s.devMenuOpen);
  const setDevMenuOpen = useStore((s) => s.setDevMenuOpen);
  const systemLogs = useStore((s) => s.systemLogs);
  const clearSystemLogs = useStore((s) => s.clearSystemLogs);
  const [copied, setCopied] = useState(false);
  const [activeFilter, setActiveFilter] = useState("ALL");
  const [apiInputUrl, setApiInputUrl] = useState(getApiBase());

  useEffect(() => {
    if (devMenuOpen) setApiInputUrl(getApiBase());
  }, [devMenuOpen]);

  const handleCopy = () => {
    const text = systemLogs.map((l) => `[${l.timestamp}] [${l.category}] ${l.message}`).join("\n");
    navigator.clipboard?.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const filteredLogs = useMemo(() => {
    if (activeFilter === "ALL") return systemLogs;
    return systemLogs.filter((l) => l.category === activeFilter);
  }, [systemLogs, activeFilter]);

  if (!devMenuOpen) return null;

  const categories = ["ALL", "UI", "API", "ENGINE", "STATE"];

  const logCatColor = (cat) => {
    if (cat === "API") return "var(--cyan)";
    if (cat === "ENGINE") return "var(--purple)";
    if (cat === "STATE") return "var(--green)";
    if (cat === "UI") return "var(--orange)";
    return "var(--t2)";
  };

  return (
    <div className="dev-menu-overlay" onClick={() => setDevMenuOpen(false)}>
      <div className="dev-menu-panel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="dev-menu-header">
          <div className="dev-menu-title">
            <Terminal size={16} color="var(--cyan)" />
            <span>Developer Console</span>
            <span style={{ fontSize: "0.65rem", color: "var(--t3)", fontWeight: 400, fontFamily: "var(--font-mono)" }}>
              {filteredLogs.length} entries
            </span>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button
              className="icon-btn"
              onClick={handleCopy}
              title="Copy all logs"
              style={{ color: copied ? "var(--green)" : undefined }}
            >
              {copied ? <Check size={15} /> : <Copy size={15} />}
            </button>
            <button className="icon-btn" onClick={clearSystemLogs} title="Clear logs">
              <Trash2 size={15} />
            </button>
            <button className="icon-btn" onClick={() => setDevMenuOpen(false)}>
              <X size={15} />
            </button>
          </div>
        </div>

        <div className="dev-menu-body">
          {/* Network settings */}
          <div className="dev-network-row">
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: "0.65rem", color: "var(--t3)", marginBottom: 6, display: "flex", alignItems: "center", gap: 6, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>
                <Network size={11} />
                FastAPI Endpoint
              </div>
              <input
                type="text"
                className="dev-network-input"
                value={apiInputUrl}
                onChange={(e) => setApiInputUrl(e.target.value)}
                placeholder="http://127.0.0.1:58102"
              />
            </div>
            <button
              style={{ padding: "8px 16px", background: "var(--cyan-dim)", border: "1px solid var(--cyan-border)", borderRadius: "var(--r-sm)", color: "var(--cyan)", fontSize: "0.78rem", fontWeight: 600, cursor: "pointer", flexShrink: 0 }}
              onClick={() => { setApiBase(apiInputUrl); window.location.reload(); }}
            >
              Reconnect
            </button>
          </div>

          {/* Hardware settings */}
          <div className="dev-network-row" style={{ marginTop: 12 }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: "0.65rem", color: "var(--t3)", marginBottom: 6, display: "flex", alignItems: "center", gap: 6, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>
                <Terminal size={11} />
                Hardware Configuration
              </div>
              <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                <select 
                  className="dev-network-input" 
                  value={embedderDevice} 
                  onChange={(e) => setEmbedderDevice(e.target.value)}
                  style={{ width: "auto", flex: 1, minWidth: "120px" }}
                >
                  <option value="auto">Embedder: Auto</option>
                  <option value="cuda">Embedder: CUDA</option>
                  <option value="cpu">Embedder: CPU</option>
                  <option value="mps">Embedder: MPS</option>
                </select>
                <select 
                  className="dev-network-input" 
                  value={llmDevice} 
                  onChange={(e) => setLlmDevice(e.target.value)}
                  style={{ width: "auto", flex: 1, minWidth: "120px" }}
                >
                  <option value="auto">LLM: Auto</option>
                  <option value="cuda">LLM: CUDA</option>
                  <option value="cpu">LLM: CPU</option>
                  <option value="mps">LLM: MPS</option>
                </select>
              </div>
            </div>
            <button
              style={{ padding: "8px 16px", background: "var(--purple-dim)", border: "1px solid var(--purple)", borderRadius: "var(--r-sm)", color: "var(--purple-light)", fontSize: "0.78rem", fontWeight: 600, cursor: "pointer", flexShrink: 0 }}
              onClick={onHardwareApply}
            >
              Apply HW
            </button>
          </div>

          {/* Category filters */}
          <div className="filter-pills">
            {categories.map((cat) => (
              <button
                key={cat}
                className={`filter-pill ${activeFilter === cat ? "active" : ""}`}
                onClick={() => setActiveFilter(cat)}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Log list */}
          <div className="log-list">
            {filteredLogs.length === 0 ? (
              <div style={{ color: "var(--t3)", fontSize: "0.72rem", textAlign: "center", padding: "20px 0" }}>
                No logs to display.
              </div>
            ) : (
              [...filteredLogs].reverse().map((log, i) => (
                <div key={i} className="log-line">
                  <span style={{ color: "var(--t3)", flexShrink: 0 }}>{log.timestamp}</span>
                  <span style={{ color: logCatColor(log.category), flexShrink: 0, fontWeight: 600, fontSize: "0.65rem", minWidth: 52 }}>
                    [{log.category}]
                  </span>
                  <span style={{ color: "var(--t2)", wordBreak: "break-word" }}>{log.message}</span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="dev-menu-footer">
          <span style={{ fontSize: "0.65rem", color: "var(--t3)", fontFamily: "var(--font-mono)" }}>
            ⬡ NSTL Dev Console · Logs are session-only
          </span>
          <button
            className="filter-pill"
            onClick={clearSystemLogs}
            style={{ fontSize: "0.65rem" }}
          >
            Clear all
          </button>
        </div>
      </div>
    </div>
  );
}
