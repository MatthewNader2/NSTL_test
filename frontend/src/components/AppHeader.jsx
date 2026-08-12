import { useState, useRef, useEffect, useMemo } from "react";
import { useStore } from "../store";
import { Zap, Cpu, Monitor, Settings, ChevronDown, Check, UserCircle } from "lucide-react";

const PROFILES = [
  { value: "A", label: "Profile A", desc: "Embedding Only" },
  { value: "B", label: "Profile B", desc: "LLM (Embedding + Text)" },
  { value: "C", label: "Profile C", desc: "Embedder + LLM (Hybrid)" },
  // Q-4 fix: Profile D was implemented in inference.py but was missing from the UI picker
  { value: "D", label: "Profile D", desc: "Embedder + LLM (No Synthesis)" },
];

function ModelPickerPopover({ models, selected, onSelect, accentClass, onClose }) {
  const ref = useRef(null);

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) onClose(); };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []); // onClose identity doesn't matter — it closes the popover which unmounts this component

  return (
    <div className="model-popover" ref={ref}>
      <div className="model-popover-title">Select Option</div>
      {models.length === 0 ? (
        <div style={{ padding: "8px", fontSize: "0.75rem", color: "var(--t3)" }}>No items found</div>
      ) : (
        models.map((m) => {
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
                <div className="model-option-name" style={{ color: isSelected ? (accentClass === "purple" ? "#a78bfa" : "#00d4ff") : "var(--t1)", wordBreak: "break-all" }}>
                  {m.label}
                </div>
                {m.desc && <div className="model-option-desc">{m.desc}</div>}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}

export default function AppHeader({ embedderModel, llmModel, onModelSwap, availableModels, selectedProfile, onProfileChange }) {
  const apiStatus = useStore((s) => s.apiStatus);
  const cells = useStore((s) => s.cells);
  const hardwareDevice = useStore((s) => s.hardwareDevice);
  const devMenuOpen = useStore((s) => s.devMenuOpen);
  const setDevMenuOpen = useStore((s) => s.setDevMenuOpen);
  const logSystemEvent = useStore((s) => s.logSystemEvent);

  const [embedderOpen, setEmbedderOpen] = useState(false);
  const [llmOpen, setLlmOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);

  // Dynamic models derived from availableModels prop
  const embedderOptions = useMemo(() => {
    if (!availableModels?.embedders) return [];
    return availableModels.embedders.map(m => ({ value: m, label: m }));
  }, [availableModels]);

  const llmOptions = useMemo(() => {
    if (!availableModels?.llms) return [];
    return availableModels.llms.map(m => ({ value: m, label: m }));
  }, [availableModels]);

  const currentEmbedder = embedderOptions.find((m) => m.value === embedderModel) || { label: embedderModel };
  const currentLlm = llmOptions.find((m) => m.value === llmModel) || { label: llmModel };
  const currentProfile = PROFILES.find((p) => p.value === selectedProfile);

  const isOnline = apiStatus === "live";
  const isGpu = hardwareDevice === "cuda" || hardwareDevice === "mps";

  return (
    <header className="app-header">
      {/* Logo */}
      <div className="header-logo">
        <span className="header-logo-mark">⬡ NSTL</span>
        {/* H-7 fix: read version from build-time env var so it stays in sync with package.json */}
        <span className="header-logo-ver">{import.meta.env.VITE_APP_VERSION ?? "v2.1"}</span>
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
        {/* Profile */}
        <div style={{ position: "relative" }}>
          <button
            className="model-badge"
            style={{ borderColor: "var(--border)", color: "var(--t1)" }}
            onClick={() => { setProfileOpen(!profileOpen); setEmbedderOpen(false); setLlmOpen(false); }}
          >
            <UserCircle size={12} style={{ color: "var(--t2)" }} />
            <span className="model-badge-label">PROF</span>
            <span style={{ fontWeight: 600 }}>{currentProfile?.value || selectedProfile}</span>
            <ChevronDown size={12} style={{ opacity: 0.6, transition: "transform 0.2s", transform: profileOpen ? "rotate(180deg)" : "rotate(0deg)" }} />
          </button>
          {profileOpen && (
            <ModelPickerPopover
              models={PROFILES}
              selected={selectedProfile}
              accentClass="cyan"
              onSelect={(v) => { onProfileChange(v); setProfileOpen(false); }}
              onClose={() => setProfileOpen(false)}
            />
          )}
        </div>

        <div style={{ width: "1px", height: "16px", background: "var(--border)" }} />

        {/* Embedder — shown for profiles with dedicated embedder (A, C, D) */}
        {(selectedProfile === "A" || selectedProfile === "C" || selectedProfile === "D") && (
          <div style={{ position: "relative" }}>
            <button
              className="model-badge cyan"
              onClick={() => { setEmbedderOpen(!embedderOpen); setLlmOpen(false); setProfileOpen(false); logSystemEvent("Opened embedder picker", "UI"); }}
            >
              <span className="badge-dot" />
              <span className="model-badge-label">EMB</span>
              <span style={{ fontWeight: 600, maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis" }}>{currentEmbedder.label}</span>
              <ChevronDown size={12} style={{ opacity: 0.6, transition: "transform 0.2s", transform: embedderOpen ? "rotate(180deg)" : "rotate(0deg)" }} />
            </button>
            {embedderOpen && (
              <ModelPickerPopover
                models={embedderOptions}
                selected={embedderModel}
                accentClass="cyan"
                onSelect={(v) => onModelSwap(v, llmModel)}
                onClose={() => setEmbedderOpen(false)}
              />
            )}
          </div>
        )}

        {/* B-9 fix: divider should only show when BOTH pickers are visible, which is only Profile C (and D) */}
        {(selectedProfile === "C" || selectedProfile === "D") && (
          <div style={{ width: "1px", height: "16px", background: "var(--border)" }} />
        )}

        {/* LLM — shown for profiles with an LLM (B, C, D) */}
        {(selectedProfile === "B" || selectedProfile === "C" || selectedProfile === "D") && (
          <div style={{ position: "relative" }}>
            <button
              className="model-badge purple"
              onClick={() => { setLlmOpen(!llmOpen); setEmbedderOpen(false); setProfileOpen(false); logSystemEvent("Opened LLM picker", "UI"); }}
            >
              <span className="badge-dot" />
              <span className="model-badge-label">LLM</span>
              <span style={{ fontWeight: 600, maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis" }}>{currentLlm.label}</span>
              <ChevronDown size={12} style={{ opacity: 0.6, transition: "transform 0.2s", transform: llmOpen ? "rotate(180deg)" : "rotate(0deg)" }} />
            </button>
            {llmOpen && (
              <ModelPickerPopover
                models={llmOptions}
                selected={llmModel}
                accentClass="purple"
                onSelect={(v) => onModelSwap(embedderModel, v)}
                onClose={() => setLlmOpen(false)}
              />
            )}
          </div>
        )}
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
