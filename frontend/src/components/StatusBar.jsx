import { useStore } from "../store";
import { Activity, GitBranch, Clock } from "lucide-react";

export default function StatusBar() {
  const logs = useStore((s) => s.logs);
  const activePath = useStore((s) => s.activePath);
  const apiStatus = useStore((s) => s.apiStatus);
  const hardwareDevice = useStore((s) => s.hardwareDevice);

  const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  return (
    <div className="status-bar">
      <div className="status-bar-left">
        <div className="status-item">
          <GitBranch size={10} />
          <span>Path: {activePath.length} cells</span>
        </div>
        <div className="status-item">
          <Activity size={10} />
          <span>{logs.length} log entries</span>
        </div>
      </div>
      <div className="status-bar-right">
        <span>WebGL · {hardwareDevice?.toUpperCase() || "CPU"}</span>
        <span style={{ color: apiStatus === "live" ? "var(--green)" : "var(--red)" }}>
          {apiStatus === "live" ? "● CONNECTED" : "○ OFFLINE"}
        </span>
        <div className="status-item">
          <Clock size={10} />
          <span>{now}</span>
        </div>
      </div>
    </div>
  );
}
