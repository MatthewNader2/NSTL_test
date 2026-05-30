import { create } from "zustand";

export const useStore = create((set) => ({
  cells: [],
  setCells: (cells) => {
    set({ cells });
    useStore.getState().logSystemEvent(`Loaded ${cells.length} semantic cells`, "ENGINE");
  },
  activePath: [],
  setActivePath: (path) => set({ activePath: path }),
  virtualEdges: new Set(),
  setVirtualEdges: (edges) => set({ virtualEdges: edges }),
  generatedCode: "# Generated code will appear here\n",
  setGeneratedCode: (code) => set({ generatedCode: code }),
  
  // Execution logs
  logs: [],
  setLogs: (logs) => set({ logs }),
  addLog: (log) => set((state) => ({ logs: [...state.logs.slice(-200), log] })),
  
  selectedNode: null,
  setSelectedNode: (node) => {
    set({ selectedNode: node });
    if (node) {
      useStore.getState().logSystemEvent(`Selected node: ${node.cell_id}`, "UI");
    }
  },
  hoveredNode: null,
  setHoveredNode: (node) => set({ hoveredNode: node }),
  apiStatus: "connecting",
  setApiStatus: (status) => {
    set({ apiStatus: status });
    useStore.getState().logSystemEvent(`API Base URL status: ${status}`, "API");
  },
  rightActiveTab: "code",
  setRightActiveTab: (tab) => set({ rightActiveTab: tab }),

  // 💬 Clickable Chat History
  chatHistory: [],
  addHistoryItem: (item) => set((state) => {
    const updatedHistory = [...state.chatHistory, item];
    useStore.getState().logSystemEvent(`Added query history item: "${item.prompt}"`, "STATE");
    return { chatHistory: updatedHistory };
  }),
  updateHistoryItem: (id, updatedFields) => set((state) => ({
    chatHistory: state.chatHistory.map((item) => item.id === id ? { ...item, ...updatedFields } : item)
  })),
  activeHistoryId: null,
  setActiveHistoryId: (id) => set({ activeHistoryId: id }),

  // 🛠️ Developer System Logs
  systemLogs: [],
  logSystemEvent: (message, category = "SYSTEM") => set((state) => {
    const timestamp = new Date().toLocaleTimeString();
    const newLog = { message, category, timestamp };
    return { systemLogs: [...state.systemLogs.slice(-499), newLog] };
  }),
  clearSystemLogs: () => set({ systemLogs: [] }),

  devMenuOpen: false,
  setDevMenuOpen: (open) => {
    set({ devMenuOpen: open });
    useStore.getState().logSystemEvent(`Toggled dev menu to: ${open}`, "UI");
  },
}));
