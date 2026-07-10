import { useState, useRef, useEffect } from "react";
import { useStore } from "../store";
import { Zap, Cpu, Monitor, Settings, ChevronDown, Check } from "lucide-react";

const EMBEDDER_MODELS = [
  { value: "jina-embeddings-v5-text-nano", label: "Jina v5 Nano", desc: "Best quality · GPU optimized" },
  { value: "jinaai/jina-embeddings-v2-small-en", label: "Jina v2 Small", desc: "Lighter weight · CPU friendly" },
  { value: "BAAI/bge-small-en-v1.5", label: "BGE Small v1.5", desc: "Fast · BAAI open-source" },
];

const LLM_MODELS = [
  { value: "qwen2.5-coder-1.5b-instruct", label: "Qwen2.5 1.5B", desc: "Higher quality · Slower" },
  { value: "qwen2.5-coder-0.5b-instruct", label: "Qwen2.5 0.5B", desc: "Fastest · Lighter" },
];

function ModelPickerPopover({ models, selected, onSelect, accentClass, onClose }) {
  const ref = useRef(null);

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) onClose(); };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  return (
    <div className="model-popover" ref={ref}>
      <div className="model-popover-title">Select Model</div>
      {models.map((m) => {
        const isSelected = selected === m.value;
        return (
          <div
            key={m.value}
            className={`model-option ${isSelected ? (accentClass === "purple" ? "selected-purple" : "selected") : ""}`}
            onClick={() => { onSelect(m.value); onClose(); }}
          >
            <div className="model-option-check">
              {isSelected && <Check size={12} color={accentClass === "purple" ? "#a78bfa" : "#00d4ff"} />}
            </div>
            <div className="model-option-info">
              <div className="model-option-name" style={{ color: isSelected ? (accentClass === "purple" ? "#a78bfa" : "#00d4ff") : "var(--t1)" }}>
                {m.label}
              </div>
              <div className="model-option-desc">{m.desc}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function AppHeader({ embedderModel, llmModel, onModelSwap }) {
  const apiStatus = useStore((s) => s.apiStatus);
  const cells = useStore((s) => s.cells);
  const hardwareDevice = useStore((s) => s.hardwareDevice);
  const devMenuOpen = useStore((s) => s.devMenuOpen);
  const setDevMenuOpen = useStore((s) => s.setDevMenuOpen);
  const logSystemEvent = useStore((s) => s.logSystemEvent);

  const [embedderOpen, setEmbedderOpen] = useState(false);
  const [llmOpen, setLlmOpen] = useState(false);

  const currentEmbedder = EMBEDDER_MODELS.find((m) => m.value === embedderModel);
  const currentLlm = LLM_MODELS.find((m) => m.value === llmModel);

  const isOnline = apiStatus === "live";
  const isGpu = hardwareDevice === "cuda" || hardwareDevice === "mps";

  return (
    <header className="app-header">
      {/* Logo */}
      <div className="header-logo">
        <span className="header-logo-mark">⬡ NSTL</span>
        <span className="header-logo-ver">v2.1</span>
      </div>

      {/* Status indicators */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <div className={`header-status-pill ${isOnline ? "online" : "offline"}`}>
          <div className={`status-dot ${isOnline ? "online" : "offline"}`} />
          <span style={{ fontSize: "0.68rem", fontWeight: 600, color: isOnline ? "var(--green)" : "var(--red)" }}>
            {isOnline ? "ONLINE" : "OFFLINE"}
          </span>
        </div>

        <div className="header-divider" />

        <div style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "0.7rem", color: "var(--t2)" }}>
          <Cpu size={12} />
          <span>{cells.length} cells</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "0.7rem", color: isGpu ? "var(--green)" : "var(--orange)" }}>
          <Monitor size={12} />
          <span style={{ fontWeight: 600 }}>{hardwareDevice?.toUpperCase() || "CPU"}</span>
        </div>
      </div>

      {/* Model Badges */}
      <div className="header-center">
        {/* Embedder */}
        <div style={{ position: "relative" }}>
          <button
            className="model-badge cyan"
            onClick={() => { setEmbedderOpen(!embedderOpen); setLlmOpen(false); logSystemEvent("Opened embedder picker", "UI"); }}
          >
            <span className="badge-dot" />
            <span className="model-badge-label">EMB</span>
            <span style={{ fontWeight: 600 }}>{currentEmbedder?.label || embedderModel}</span>
            <ChevronDown size={12} style={{ opacity: 0.6, transition: "transform 0.2s", transform: embedderOpen ? "rotate(180deg)" : "rotate(0deg)" }} />
          </button>
          {embedderOpen && (
            <ModelPickerPopover
              models={EMBEDDER_MODELS}
              selected={embedderModel}
              accentClass="cyan"
              onSelect={(v) => onModelSwap(v, llmModel)}
              onClose={() => setEmbedderOpen(false)}
            />
          )}
        </div>

        <div style={{ width: "1px", height: "16px", background: "var(--border)" }} />

        {/* LLM */}
        <div style={{ position: "relative" }}>
          <button
            className="model-badge purple"
            onClick={() => { setLlmOpen(!llmOpen); setEmbedderOpen(false); logSystemEvent("Opened LLM picker", "UI"); }}
          >
            <span className="badge-dot" />
            <span className="model-badge-label">LLM</span>
            <span style={{ fontWeight: 600 }}>{currentLlm?.label || llmModel}</span>
            <ChevronDown size={12} style={{ opacity: 0.6, transition: "transform 0.2s", transform: llmOpen ? "rotate(180deg)" : "rotate(0deg)" }} />
          </button>
          {llmOpen && (
            <ModelPickerPopover
              models={LLM_MODELS}
              selected={llmModel}
              accentClass="purple"
              onSelect={(v) => onModelSwap(embedderModel, v)}
              onClose={() => setLlmOpen(false)}
            />
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="header-actions">
        <button
          className={`icon-btn ${devMenuOpen ? "active" : ""}`}
          onClick={() => { setDevMenuOpen(!devMenuOpen); }}
          title="Developer Console"
        >
          <Settings size={16} />
        </button>
      </div>
    </header>
  );
}
