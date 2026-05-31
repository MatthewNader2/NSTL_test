import { useStore } from "../store";
import MonacoEditor from "./MonacoEditor";
import {
  Code2,
  Eye,
  GitBranch,
  Terminal,
  X,
  Copy,
  Check,
  Download,
} from "lucide-react";
import { useState, useCallback } from "react";

export default function RightPanel({ open, isMobile, mobileActive, onToggle }) {
  const activeTab = useStore((s) => s.rightActiveTab);
  const setActiveTab = useStore((s) => s.setRightActiveTab);
  const selectedNode = useStore((s) => s.selectedNode);
  const generatedCode = useStore((s) => s.generatedCode);
  const activePath = useStore((s) => s.activePath);
  const logs = useStore((s) => s.logs);
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard?.writeText(generatedCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }, [generatedCode]);

  const handleDownload = useCallback(() => {
    const element = document.createElement("a");
    const file = new Blob([generatedCode], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = "generated_pipeline.py";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  }, [generatedCode]);

  const tabs = [
    { id: "code", icon: <Code2 size={14} />, label: "Code" },
    {
      id: "inspect",
      icon: <Eye size={14} />,
      label: selectedNode?.cell_id || "Inspect",
    },
    {
      id: "path",
      icon: <GitBranch size={14} />,
      label: `Path (${activePath.length})`,
    },
    { id: "log", icon: <Terminal size={14} />, label: "Logs" },
  ];

  return (
    <div
      className={`right-panel glass-panel ${!isMobile && !open ? "collapsed" : ""} ${isMobile && mobileActive ? "mobile-active" : ""}`}
      style={{}}
    >
      <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
        {/* Tabs Header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0.375rem 0.75rem",
            borderBottom: "1px solid var(--glass-border)",
            background: "rgba(4, 8, 20, 0.6)",
            gap: "0.5rem",
          }}
        >
          <div
            style={{
              display: "flex",
              gap: "0.375rem",
              flex: 1,
              overflowX: "auto",
            }}
          >
            {tabs.map((t) => {
              const isActive = activeTab === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => {
                    setActiveTab(t.id);
                    useStore.getState().logSystemEvent(`Switch right panel tab to: ${t.id}`, "UI");
                  }}
                  style={{
                    padding: "0.375rem 0.75rem",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.375rem",
                    borderRadius: "6px",
                    background: isActive ? "rgba(0, 229, 255, 0.12)" : "transparent",
                    border: isActive ? "1px solid rgba(0, 229, 255, 0.25)" : "1px solid transparent",
                    color: isActive ? "var(--accent)" : "var(--text-secondary)",
                    fontWeight: isActive ? "600" : "400",
                    whiteSpace: "nowrap",
                    fontSize: "0.75rem",
                    boxShadow: isActive ? "0 0 8px rgba(0, 229, 255, 0.08)" : "none",
                  }}
                >
                  {t.icon}
                  <span>{t.label}</span>
                </button>
              );
            })}
          </div>
          {/* Mobile Close Button */}
          <button
            onClick={onToggle}
            style={{
              padding: "0.375rem",
              color: "var(--text-secondary)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: "6px",
            }}
            className="mobile-close-btn"
          >
            <X size={14} />
          </button>
        </div>

        {/* Code Action Bar */}
        {activeTab === "code" && (
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              padding: "8px 12px",
              borderBottom: "1px solid var(--glass-border)",
              fontSize: "0.8rem",
              color: "var(--text-secondary)",
              background: "#111b2d",
            }}
          >
            <span>📄 generated_pipeline.py</span>
            <div style={{ display: "flex", gap: "12px" }}>
              <button
                onClick={handleDownload}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  color: "var(--text-secondary)",
                }}
              >
                <Download size={14} /> Download
              </button>
              <button
                onClick={handleCopy}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  color: copied ? "#98c379" : "var(--text-secondary)",
                }}
              >
                {copied ? <Check size={14} /> : <Copy size={14} />}{" "}
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
          </div>
        )}

        {/* Content Area */}
        <div style={{ flex: 1, overflow: "hidden" }}>
          {activeTab === "code" && (
            <div style={{ height: "100%" }}>
              <MonacoEditor />
            </div>
          )}

          {activeTab === "inspect" && selectedNode && (
            <div style={{ padding: "20px", overflowY: "auto", height: "100%" }}>
              <h2 style={{ color: "var(--accent)", marginBottom: "4px" }}>
                {selectedNode.cell_id}
              </h2>
              <p
                style={{
                  color: "var(--text-secondary)",
                  marginBottom: "16px",
                  fontSize: "0.9rem",
                }}
              >
                Stage {selectedNode.stage} Execution Node
              </p>

              <div
                style={{
                  background: "rgba(0,0,0,0.3)",
                  padding: "12px",
                  borderRadius: "8px",
                  marginBottom: "16px",
                  fontSize: "0.85rem",
                }}
              >
                <div style={{ marginBottom: "8px" }}>
                  <strong>Type Constraint:</strong>{" "}
                  {selectedNode.inputs.input_type} →{" "}
                  {selectedNode.outputs.output_type}
                </div>
                <div>
                  <strong>State Transition:</strong> [
                  {selectedNode.inputs.expected_state}] → [
                  {selectedNode.outputs.resulting_state}]
                </div>
              </div>

              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {selectedNode.keywords?.map((k) => (
                  <span
                    key={k}
                    style={{
                      background: "rgba(0,229,255,0.1)",
                      border: "1px solid rgba(0,229,255,0.3)",
                      padding: "4px 8px",
                      borderRadius: "12px",
                      fontSize: "0.75rem",
                      color: "var(--accent)",
                    }}
                  >
                    #{k}
                  </span>
                ))}
              </div>

              <div style={{ marginTop: "24px" }}>
                <h4
                  style={{
                    color: "var(--text-secondary)",
                    marginBottom: "8px",
                  }}
                >
                  Template Source Code
                </h4>
                <pre
                  style={{
                    background: "#0d1117",
                    padding: "12px",
                    borderRadius: "6px",
                    fontSize: "1.6rem",
                    overflowX: "auto",
                    border: "1px solid #30363d",
                    color: "#c9d1d9",
                  }}
                >
                  {selectedNode.code_template || "# No implementation"}
                </pre>
              </div>
            </div>
          )}

          {activeTab === "path" && (
            <div style={{ padding: "16px", overflowY: "auto", height: "100%" }}>
              {activePath.map((cell, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    padding: "12px 0",
                    borderBottom: "1px solid rgba(255,255,255,0.05)",
                  }}
                >
                  <span
                    style={{
                      color: "var(--text-secondary)",
                      fontSize: "0.9rem",
                    }}
                  >
                    {i + 1}.
                  </span>
                  <div style={{ display: "flex", flexDirection: "column" }}>
                    <span
                      style={{
                        color: "var(--accent)",
                        fontWeight: "bold",
                        fontSize: "0.95rem",
                      }}
                    >
                      {cell.cell_id}
                    </span>
                    <span
                      style={{
                        color: "var(--text-secondary)",
                        fontSize: "0.75rem",
                      }}
                    >
                      {cell.outputs.resulting_state}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === "log" && (
            <div style={{ padding: "16px", overflowY: "auto", height: "100%", fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
              {logs.length === 0 ? (
                <div style={{ color: "var(--text-secondary)", textAlign: "center", marginTop: "40px" }}>
                  No execution logs available. Run a query in the console.
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  {logs.map((log, i) => {
                    let color = "var(--text-primary)";
                    let borderLeftColor = "rgba(255,255,255,0.1)";
                    if (log.type === "system") { color = "#c678dd"; borderLeftColor = "#c678dd"; }
                    else if (log.type === "warn" || log.type === "error") { color = "#e06c75"; borderLeftColor = "#e06c75"; }
                    else if (log.type === "debug") { color = "var(--text-secondary)"; borderLeftColor = "var(--text-secondary)"; }
                    else if (log.type === "info") { color = "#61afef"; borderLeftColor = "#61afef"; }
                    
                    return (
                      <div
                        key={i}
                        style={{
                          background: "rgba(0,0,0,0.25)",
                          padding: "6px 10px",
                          borderRadius: "4px",
                          borderLeft: `3px solid ${borderLeftColor}`,
                          color: color,
                          wordBreak: "break-word",
                          lineHeight: "1.3"
                        }}
                      >
                        <span style={{ opacity: 0.6, fontSize: "0.65rem", marginRight: "6px" }}>
                          [{log.type?.toUpperCase()}]
                        </span>
                        {log.msg}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Toggle button when collapsed */}
      {!open && (
        <button
          onClick={onToggle}
          style={{
            position: "absolute",
            top: "0.5rem",
            right: "0.5rem",
            zIndex: 10,
            color: "var(--text-secondary)",
          }}
          title="Expand Panel"
        >
          <Code2 size={16} />
        </button>
      )}
    </div>
  );
}
