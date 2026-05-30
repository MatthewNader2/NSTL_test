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

export default function RightPanel({ open, onToggle }) {
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
    <div className={`right-panel glass-panel ${open ? "" : "collapsed"}`}>
      <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
        {/* Tabs Header */}
        <div
          style={{
            display: "flex",
            borderBottom: "1px solid var(--glass-border)",
            background: "rgba(0,0,0,0.3)",
            overflowX: "auto",
          }}
        >
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              style={{
                flex: 1,
                padding: "10px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 6,
                color:
                  activeTab === t.id
                    ? "var(--accent)"
                    : "var(--text-secondary)",
                borderBottom:
                  activeTab === t.id
                    ? "2px solid var(--accent)"
                    : "2px solid transparent",
                fontWeight: activeTab === t.id ? "bold" : "normal",
                whiteSpace: "nowrap",
              }}
            >
              {t.icon} <span style={{ fontSize: "0.85rem" }}>{t.label}</span>
            </button>
          ))}
          {/* Mobile Close Button */}
          <button
            onClick={onToggle}
            style={{ padding: "0 15px", color: "var(--text-secondary)" }}
            className="mobile-close-btn"
          >
            <X size={18} />
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
                    fontSize: "0.8rem",
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
        </div>
      </div>
    </div>
  );
}
