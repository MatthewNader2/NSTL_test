import { useStore } from "../store";
import { Zap, Cpu, Layers, Settings } from "lucide-react";

export default function TitleBar({ onSettingsClick }) {
  const apiStatus = useStore((s) => s.apiStatus);
  const cells = useStore((s) => s.cells);

  return (
    <div
      style={{
        height: "2.25rem", // 36px in rem
        background: "rgba(4, 10, 20, 0.9)",
        borderBottom: "1px solid var(--glass-border)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 0.75rem",
        fontSize: "0.6875rem", // 11px in rem
        color: "var(--text-secondary)",
      }}
    >
      <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
        <span
          style={{ fontWeight: 700, color: "var(--accent)", letterSpacing: 2 }}
        >
          ⬡ NSTL
        </span>
        <span>v2.1</span>
        <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
          <Zap size={10} color={apiStatus === "live" ? "#98c379" : "#e06c75"} />
          {apiStatus === "live" ? "LIVE" : "OFFLINE"}
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
          <Cpu size={10} />
          {cells.length} cells
        </span>
      </div>
      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <Layers size={12} />
          <span>all‑MiniLM‑L6‑v2</span>
        </div>
        <button
          onClick={onSettingsClick}
          style={{
            display: "flex",
            alignItems: "center",
            color: "var(--text-secondary)",
            padding: "2px",
          }}
          title="Network Connection"
        >
          <Settings size={12} />
        </button>
      </div>
    </div>
  );
}
