import { useStore } from "../store";
import { Activity, GitBranch } from "lucide-react";

export default function StatusBar() {
  const logs = useStore((s) => s.logs);
  const activePath = useStore((s) => s.activePath);

  return (
    <div
      style={{
        height: "1.375rem", // 22px in rem
        background: "rgba(2, 6, 14, 0.9)",
        borderTop: "1px solid var(--glass-border)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 0.625rem",
        fontSize: "0.5625rem", // 9px in rem
        color: "var(--text-secondary)",
      }}
    >
      <div style={{ display: "flex", gap: "0.75rem" }}>
        <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
          <GitBranch size={10} /> Path: {activePath.length} cells
        </span>
        <span>{logs.length} log entries</span>
      </div>
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <span>RENDER: WebGL</span>
        <span>LATENCY: &lt;10ms</span>
      </div>
    </div>
  );
}
