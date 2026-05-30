import { useStore } from "../store";
import { X, Trash2, Copy, Check, Terminal } from "lucide-react";
import { useState, useMemo } from "react";

export default function DevMenu() {
  const devMenuOpen = useStore((s) => s.devMenuOpen);
  const setDevMenuOpen = useStore((s) => s.setDevMenuOpen);
  const systemLogs = useStore((s) => s.systemLogs);
  const clearSystemLogs = useStore((s) => s.clearSystemLogs);
  const [copied, setCopied] = useState(false);
  const [activeFilter, setActiveFilter] = useState("ALL");

  const handleCopy = () => {
    const text = systemLogs
      .map((log) => `[${log.timestamp}] [${log.category}] ${log.message}`)
      .join("\n");
    navigator.clipboard?.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const filteredLogs = useMemo(() => {
    if (activeFilter === "ALL") return systemLogs;
    return systemLogs.filter((log) => log.category === activeFilter);
  }, [systemLogs, activeFilter]);

  if (!devMenuOpen) return null;

  const categories = ["ALL", "UI", "API", "ENGINE", "STATE"];

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(3, 5, 12, 0.75)",
        backdropFilter: "blur(6px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 200000,
        fontFamily: "var(--font-mono)",
        padding: "1rem",
      }}
      onClick={() => setDevMenuOpen(false)}
    >
      <div
        className="glass-panel"
        style={{
          width: "100%",
          maxWidth: "700px",
          height: "80vh",
          border: "1px solid var(--accent)",
          boxShadow: "0 20px 50px rgba(0, 229, 255, 0.15)",
          padding: "1rem",
          display: "flex",
          flexDirection: "column",
          gap: "10px",
          background: "rgba(6, 10, 24, 0.95)",
        }}
        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking content
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            borderBottom: "1px solid var(--glass-border)",
            paddingBottom: "0.5rem",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--accent)" }}>
            <Terminal size={18} />
            <h3 style={{ fontSize: "1rem", fontWeight: "600", margin: 0 }}>⬡ Developer Log Console</h3>
          </div>
          <button
            onClick={() => setDevMenuOpen(false)}
            style={{ color: "var(--text-secondary)", display: "flex", padding: "4px" }}
          >
            <X size={16} />
          </button>
        </div>

        {/* Filters */}
        <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
          {categories.map((cat) => {
            const isSelected = activeFilter === cat;
            return (
              <button
                key={cat}
                onClick={() => setActiveFilter(cat)}
                style={{
                  fontSize: "0.65rem",
                  padding: "4px 8px",
                  borderRadius: "4px",
                  background: isSelected ? "rgba(0, 229, 255, 0.15)" : "rgba(255, 255, 255, 0.04)",
                  border: isSelected ? "1px solid var(--accent)" : "1px solid rgba(255, 255, 255, 0.08)",
                  color: isSelected ? "var(--accent)" : "var(--text-secondary)",
                  fontWeight: isSelected ? "bold" : "normal",
                }}
              >
                {cat}
              </button>
            );
          })}
        </div>

        {/* Scrollable Logs list */}
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            background: "rgba(0,0,0,0.4)",
            border: "1px solid rgba(255,255,255,0.04)",
            borderRadius: "6px",
            padding: "8px",
            display: "flex",
            flexDirection: "column",
            gap: "6px",
            fontSize: "0.7rem",
          }}
        >
          {filteredLogs.length === 0 ? (
            <div
              style={{
                flex: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "var(--text-secondary)",
                fontStyle: "italic",
              }}
            >
              No developer events recorded under filter "{activeFilter}".
            </div>
          ) : (
            filteredLogs.map((log, i) => {
              let catBg = "rgba(255,255,255,0.06)";
              let catColor = "var(--text-secondary)";
              if (log.category === "UI") { catBg = "rgba(97, 175, 239, 0.12)"; catColor = "#61afef"; }
              else if (log.category === "API") { catBg = "rgba(152, 195, 121, 0.12)"; catColor = "#98c379"; }
              else if (log.category === "ENGINE") { catBg = "rgba(198, 120, 221, 0.12)"; catColor = "#c678dd"; }
              else if (log.category === "STATE") { catBg = "rgba(224, 108, 117, 0.12)"; catColor = "#e06c75"; }

              return (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: "8px",
                    lineHeight: "1.4",
                    borderBottom: "1px solid rgba(255,255,255,0.02)",
                    paddingBottom: "4px",
                  }}
                >
                  <span style={{ color: "var(--text-secondary)", opacity: 0.5, whiteSpace: "nowrap" }}>
                    [{log.timestamp}]
                  </span>
                  <span
                    style={{
                      fontSize: "0.55rem",
                      padding: "1px 4px",
                      borderRadius: "3px",
                      background: catBg,
                      color: catColor,
                      fontWeight: "bold",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {log.category}
                  </span>
                  <span style={{ color: "var(--text-primary)", wordBreak: "break-all" }}>
                    {log.message}
                  </span>
                </div>
              );
            })
          )}
        </div>

        {/* Footer controls */}
        <div style={{ display: "flex", justifyBetween: "space-between", gap: "10px", marginTop: "4px" }}>
          <button
            onClick={clearSystemLogs}
            style={{
              padding: "6px 12px",
              background: "rgba(224, 108, 117, 0.08)",
              border: "1px solid rgba(224, 108, 117, 0.2)",
              borderRadius: "6px",
              color: "#e06c75",
              fontSize: "0.75rem",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              marginRight: "auto",
            }}
          >
            <Trash2 size={12} /> Clear Logs
          </button>

          <button
            onClick={handleCopy}
            disabled={systemLogs.length === 0}
            style={{
              padding: "6px 12px",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: "6px",
              color: "var(--text-primary)",
              fontSize: "0.75rem",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              opacity: systemLogs.length === 0 ? 0.4 : 1,
            }}
          >
            {copied ? <Check size={12} color="#98c379" /> : <Copy size={12} />}
            {copied ? "Copied!" : "Copy Logs"}
          </button>
        </div>
      </div>
    </div>
  );
}
