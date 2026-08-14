import { useStore } from "../store";
import { Zap, Cpu, Layers, Settings, Terminal, Monitor } from "lucide-react";

export default function TitleBar({ embedderModel, llmModel, onModelSwap }) {
  const apiStatus = useStore((s) => s.apiStatus);
  const cells = useStore((s) => s.cells);
  const hardwareDevice = useStore((s) => s.hardwareDevice);
  const devMenuOpen = useStore((s) => s.devMenuOpen);
  const setDevMenuOpen = useStore((s) => s.setDevMenuOpen);
  const logSystemEvent = useStore((s) => s.logSystemEvent);

  const handleEmbedderChange = (e) => {
    logSystemEvent(`Embedder model dropdown changed to: ${e.target.value}`, "UI");
    onModelSwap(e.target.value, llmModel);
  };

  const handleLlmChange = (e) => {
    logSystemEvent(`LLM model dropdown changed to: ${e.target.value}`, "UI");
    onModelSwap(embedderModel, e.target.value);
  };

  const selectStyle = {
    background: "rgba(0, 0, 0, 0.4)",
    border: "1px solid rgba(0, 229, 255, 0.15)",
    borderRadius: "4px",
    color: "var(--accent)",
    fontSize: "0.65rem",
    padding: "2px 6px",
    outline: "none",
    cursor: "pointer",
    appearance: "none",
    fontWeight: "bold",
    fontFamily: "var(--font-mono)"
  };

  return (
    <div
      style={{
        height: "2.5rem",
        background: "linear-gradient(90deg, rgba(4,10,20,0.95) 0%, rgba(8,16,30,0.95) 100%)",
        borderBottom: "1px solid var(--glass-border)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 1rem",
        fontSize: "0.7rem",
        color: "var(--text-secondary)",
        userSelect: "none",
        boxShadow: "0 4px 12px rgba(0,0,0,0.2)"
      }}
    >
      <div style={{ display: "flex", gap: "1.2rem", alignItems: "center" }}>
        <span
          style={{ fontWeight: 800, color: "var(--accent)", letterSpacing: 2, textShadow: "0 0 10px rgba(0, 229, 255, 0.3)" }}
        >
          ⬡ NSTL
        </span>
        <span style={{ opacity: 0.6 }}>v2.1</span>
        
        {/* Connection Status */}
        <div style={{ display: "flex", alignItems: "center", gap: "6px", background: "rgba(255,255,255,0.03)", padding: "2px 8px", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.05)" }}>
          <Zap size={10} color={apiStatus === "live" ? "#00e5ff" : "#e06c75"} style={{ filter: `drop-shadow(0 0 4px ${apiStatus === "live" ? "#00e5ff" : "#e06c75"})` }} />
          <span style={{ color: apiStatus === "live" ? "#00e5ff" : "#e06c75", fontWeight: "bold", fontSize: "0.65rem" }}>
            {apiStatus === "live" ? "ONLINE" : "OFFLINE"}
          </span>
        </div>
        
        {/* Cell Count */}
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <Cpu size={12} color="var(--text-secondary)" />
          <span>{cells.length} cells</span>
        </div>
        
        {/* Hardware Device */}
        <div style={{ display: "flex", alignItems: "center", gap: "6px", color: hardwareDevice === "cuda" || hardwareDevice === "mps" ? "#98c379" : "#e5c07b" }}>
          <Monitor size={12} />
          <span style={{ fontWeight: "bold" }}>{hardwareDevice?.toUpperCase() || "CPU"}</span>
        </div>
      </div>

      <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
        
        {/* Embedder Select */}
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <Layers size={12} color="var(--accent)" />
          <select value={embedderModel} onChange={handleEmbedderChange} style={selectStyle}>
            <option value="jina-embeddings-v5-text-nano">Jina v5 Nano (Embedder)</option>
            <option value="jina-embeddings-v5-text-small">Jina v5 Small (Embedder)</option>
            <option value="Qwen3-Embedding-0.6B-GGUF">Qwen3 0.6B GGUF (Embedder)</option>
            <option value="jinaai/jina-embeddings-v2-small-en">Jina v2 Small (Embedder)</option>
            <option value="BAAI/bge-small-en-v1.5">BGE Small v1.5 (Embedder)</option>
          </select>
        </div>

        {/* LLM Select */}
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <Terminal size={12} color="#c678dd" />
          <select value={llmModel} onChange={handleLlmChange} style={{ ...selectStyle, color: "#c678dd", borderColor: "rgba(198, 120, 221, 0.2)" }}>
            <option value="qwen2.5-coder-1.5b-instruct">Qwen2.5 1.5B (LLM)</option>
            <option value="qwen2.5-coder-0.5b-instruct">Qwen2.5 0.5B (LLM)</option>
          </select>
        </div>

        <div style={{ width: "1px", height: "16px", background: "var(--glass-border)", margin: "0 4px" }} />

        {/* DevMenu Toggle */}
        <button
          onClick={() => {
            setDevMenuOpen(!devMenuOpen);
            if (!devMenuOpen) logSystemEvent("Opened Developer Console", "UI");
          }}
          style={{
            display: "flex",
            alignItems: "center",
            color: devMenuOpen ? "var(--accent)" : "var(--text-secondary)",
            padding: "4px",
            background: devMenuOpen ? "rgba(0, 229, 255, 0.1)" : "transparent",
            border: "1px solid",
            borderColor: devMenuOpen ? "rgba(0, 229, 255, 0.2)" : "transparent",
            borderRadius: "4px",
            transition: "all 0.2s"
          }}
          title="Developer Network & Logs Console"
        >
          <Settings size={14} />
        </button>
      </div>
    </div>
  );
}
