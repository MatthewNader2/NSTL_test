import { useStore } from "../store";
import { Code2, Box, GitBranch, Terminal, Copy, Check, Download, Eye } from "lucide-react";
import { Suspense, lazy, useState, useCallback } from "react";
import ThreeScene from "./ThreeScene";

const MonacoEditor = lazy(() => import("./MonacoEditor"));

const TABS = [
  { id: "code", icon: Code2, label: "Code Output" },
  { id: "lattice", icon: Box, label: "3D Lattice" },
  { id: "inspect", icon: Eye, label: "Inspector" },
  { id: "path", icon: GitBranch, label: "Path" },
  { id: "log", icon: Terminal, label: "Logs" },
];

export default function MainArea({ className = "" }) {
  const [activeTab, setActiveTab] = useState("code");
  const [copied, setCopied] = useState(false);

  const selectedNode = useStore((s) => s.selectedNode);
  const generatedCode = useStore((s) => s.generatedCode);
  const activePath = useStore((s) => s.activePath);
  const logs = useStore((s) => s.logs);
  const logSystemEvent = useStore((s) => s.logSystemEvent);

  const handleCopy = useCallback(() => {
    navigator.clipboard?.writeText(generatedCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }, [generatedCode]);

  const handleDownload = useCallback(() => {
    const a = document.createElement("a");
    const blobUrl = URL.createObjectURL(new Blob([generatedCode], { type: "text/plain" }));
    a.href = blobUrl;
    a.download = "generated_pipeline.py";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(blobUrl);
  }, [generatedCode]);

  const switchTab = (id) => {
    setActiveTab(id);
    logSystemEvent(`Tab switched to: ${id}`, "UI");
  };

  return (
    <main className={`main-area ${className}`}>
      {/* Tab bar */}
      <div className="tab-bar">
        {TABS.map((t) => {
          const Icon = t.icon;
          const isActive = activeTab === t.id;
          // Badge counts
          let badge = null;
          if (t.id === "path" && activePath.length > 0) badge = activePath.length;
          if (t.id === "log" && logs.length > 0) badge = logs.length;

          return (
            <button
              key={t.id}
              className={`tab-btn ${isActive ? "active" : ""}`}
              onClick={() => switchTab(t.id)}
            >
              <Icon size={14} />
              <span>{t.label}</span>
              {badge !== null && <span className="tab-badge">{badge}</span>}
            </button>
          );
        })}

        <div className="tab-spacer" />

        {/* Tab-level actions */}
        {activeTab === "code" && (
          <div className="tab-actions">
            <button className="toolbar-action-btn" onClick={handleDownload}>
              <Download size={13} />
              <span>Download</span>
            </button>
            <button className="toolbar-action-btn" onClick={handleCopy} style={{ color: copied ? "var(--green)" : undefined }}>
              {copied ? <Check size={13} /> : <Copy size={13} />}
              <span>{copied ? "Copied!" : "Copy"}</span>
            </button>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="tab-content">
        {/* Code tab */}
        {activeTab === "code" && (
          <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
            <div className="code-toolbar">
              <div className="code-toolbar-file">
                <Code2 size={13} />
                <span>generated_pipeline.py</span>
              </div>
            </div>
            <div style={{ flex: 1, overflow: "hidden" }}>
              <Suspense fallback={<div style={{ color: "var(--t2)", padding: 24, textAlign: "center" }}>Loading editor…</div>}>
                <MonacoEditor />
              </Suspense>
            </div>
          </div>
        )}

        {/* 3D Lattice tab */}
        {activeTab === "lattice" && (
          <div style={{ width: "100%", height: "100%", position: "relative" }}>
            <ThreeScene />
          </div>
        )}

        {/* Inspector tab */}
        {activeTab === "inspect" && (
          <div className="inspect-view">
            {selectedNode ? (
              <>
                <h2 style={{ color: "var(--cyan)", marginBottom: 4, fontFamily: "var(--font-display)", fontSize: "1.2rem" }}>
                  {selectedNode.cell_id}
                </h2>
                <p style={{ color: "var(--t2)", fontSize: "0.82rem", marginBottom: 16 }}>
                  Stage {selectedNode.stage} · {selectedNode.inputs?.type_name} → {selectedNode.outputs?.type_name}
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 20 }}>
                  {selectedNode.keywords?.map((k) => (
                    <span key={k} className="inspect-tag">#{k}</span>
                  ))}
                </div>
                <div style={{ fontSize: "0.75rem", color: "var(--t2)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>State Transition</div>
                <div style={{ background: "var(--bg-2)", borderRadius: "var(--r-sm)", padding: "10px 14px", fontSize: "0.8rem", marginBottom: 20, border: "1px solid var(--border)" }}>
                  [{selectedNode.inputs?.state}] → [{selectedNode.outputs?.state}]
                </div>
                <div style={{ fontSize: "0.75rem", color: "var(--t2)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>Template Source</div>
                <pre style={{ background: "#0d1117", border: "1px solid #21262d", borderRadius: "var(--r-sm)", padding: 14, fontSize: "0.75rem", overflow: "auto", color: "#c9d1d9" }}>
                  {selectedNode.code_template || "# No implementation"}
                </pre>
              </>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 12, color: "var(--t2)", textAlign: "center" }}>
                <Eye size={28} style={{ opacity: 0.4 }} />
                <div style={{ fontSize: "0.85rem" }}>Click a node in the 3D Lattice to inspect it here.</div>
              </div>
            )}
          </div>
        )}

        {/* Path tab */}
        {activeTab === "path" && (
          <div className="path-view">
            {activePath.length === 0 ? (
              <div style={{ color: "var(--t2)", textAlign: "center", marginTop: 40, fontSize: "0.85rem" }}>
                Run a query to see the compiled execution path.
              </div>
            ) : (
              activePath.map((cell, i) => (
                <div key={i} className="path-cell-row">
                  <span style={{ color: "var(--t3)", minWidth: 24, fontSize: "0.75rem" }}>{i + 1}.</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ color: "var(--cyan)", fontWeight: 600, fontSize: "0.85rem" }}>{cell.cell_id}</div>
                    <div style={{ color: "var(--t2)", fontSize: "0.72rem" }}>{cell.outputs?.state}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {/* Log tab */}
        {activeTab === "log" && (
          <div className="log-view">
            {logs.length === 0 ? (
              <div style={{ color: "var(--t2)", textAlign: "center", marginTop: 40, fontSize: "0.85rem" }}>
                No execution logs yet. Run a query first.
              </div>
            ) : (
              logs.map((log, i) => {
                const type = log.type || "debug";
                return (
                  <div key={i} className={`log-entry ${type}`}>
                    <span style={{ opacity: 0.5, fontSize: "0.65rem", marginRight: 8 }}>[{type.toUpperCase()}]</span>
                    {log.msg}
                  </div>
                );
              })
            )}
          </div>
        )}
      </div>
    </main>
  );
}
