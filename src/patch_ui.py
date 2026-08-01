import sys

with open("frontend/src/hooks/useApi.js", "r", encoding="utf-8") as f:
    code = f.read()
code = code.replace("export async function initializeEngine(profile, device) {", "export async function initializeEngine(profile, embedder_model, llm_model, embedder_device, llm_device, trees_storage) {")
code = code.replace("body: JSON.stringify({ profile, device }),", "body: JSON.stringify({ profile, embedder_model, llm_model, embedder_device, llm_device, trees_storage }),")
with open("frontend/src/hooks/useApi.js", "w", encoding="utf-8") as f:
    f.write(code)

with open("frontend/src/App.jsx", "r", encoding="utf-8") as f:
    app_code = f.read()

# Add state variables
target_state = 'const [selectedDevice, setSelectedDevice] = useState("auto");'
replacement_state = '''const [embedderModel, setEmbedderModel] = useState("jinaai/jina-embeddings-v2-small-en");
  const [llmModel, setLlmModel] = useState("qwen2.5-coder-0.5b-instruct");
  const [embedderDevice, setEmbedderDevice] = useState("auto");
  const [llmDevice, setLlmDevice] = useState("auto");
  const [treesStorage, setTreesStorage] = useState("ram");'''
app_code = app_code.replace(target_state, replacement_state)

# Replace the hardware profile UI (approx lines 270-320)
target_ui = '''          {/* Hardware Profile */}
          <div className="mb-4">
            <label className="block text-gray-400 text-sm font-bold mb-2 uppercase tracking-wide">
              Hardware Profile
            </label>
            <select
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-white focus:outline-none focus:border-cyan-400 transition-colors"
              value={selectedProfile}
              onChange={(e) => setSelectedProfile(e.target.value)}
            >
              <option value="A">Profile A (Fast SentenceTransformer)</option>
              <option value="B">Profile B (Local LLM Qwen0.5b)</option>
              <option value="C">Profile C (Unloaded)</option>
            </select>
          </div>

          {/* Hardware Override */}
          <div className="mb-6">
            <label className="block text-gray-400 text-sm font-bold mb-2 uppercase tracking-wide">
              Hardware Override
            </label>
            <select
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-white focus:outline-none focus:border-cyan-400 transition-colors"
              value={selectedDevice}
              onChange={(e) => setSelectedDevice(e.target.value)}
            >
              <option value="auto">Auto (HardwareProfiler)</option>
              <option value="cpu">CPU Only</option>
              <option value="cuda">NVIDIA CUDA</option>
              <option value="mps">Apple Metal (MPS)</option>
            </select>
          </div>'''

replacement_ui = '''          {/* Hardware Profile */}
          <div className="mb-4">
            <label className="block text-gray-400 text-sm font-bold mb-2 uppercase tracking-wide">
              Pipeline Profile
            </label>
            <select
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-white focus:outline-none focus:border-cyan-400 transition-colors"
              value={selectedProfile}
              onChange={(e) => setSelectedProfile(e.target.value)}
            >
              <option value="A">Profile A (Embedding Only)</option>
              <option value="B">Profile B (LLM Only + Feedback)</option>
              <option value="C">Profile C (Hybrid + Feedback)</option>
            </select>
          </div>

          <div className="mb-4">
            <label className="block text-gray-400 text-sm font-bold mb-2 uppercase tracking-wide">Embedder Model</label>
            <input type="text" className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-white" value={embedderModel} onChange={(e) => setEmbedderModel(e.target.value)} />
          </div>
          
          <div className="mb-4">
            <label className="block text-gray-400 text-sm font-bold mb-2 uppercase tracking-wide">LLM Model (Folder Name)</label>
            <input type="text" className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-white" value={llmModel} onChange={(e) => setLlmModel(e.target.value)} />
          </div>

          <div className="flex gap-4 mb-6">
            <div className="flex-1">
              <label className="block text-gray-400 text-sm font-bold mb-2 uppercase tracking-wide">Embedder Device</label>
              <select className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-white" value={embedderDevice} onChange={(e) => setEmbedderDevice(e.target.value)}>
                <option value="auto">Auto</option><option value="cpu">CPU</option><option value="cuda">CUDA</option>
              </select>
            </div>
            <div className="flex-1">
              <label className="block text-gray-400 text-sm font-bold mb-2 uppercase tracking-wide">LLM Device</label>
              <select className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-white" value={llmDevice} onChange={(e) => setLlmDevice(e.target.value)}>
                <option value="auto">Auto</option><option value="cpu">CPU</option><option value="cuda">CUDA</option>
              </select>
            </div>
            <div className="flex-1">
              <label className="block text-gray-400 text-sm font-bold mb-2 uppercase tracking-wide">Trees Storage</label>
              <select className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-white" value={treesStorage} onChange={(e) => setTreesStorage(e.target.value)}>
                <option value="ram">RAM</option><option value="vram">VRAM (GPU)</option>
              </select>
            </div>
          </div>'''
app_code = app_code.replace(target_ui, replacement_ui)

# Update logSystemEvent and initializeEngine call
app_code = app_code.replace('logSystemEvent(`Applying settings: URL=${apiInputUrl}, Profile=${selectedProfile}, Device=${selectedDevice}`, "API");', 'logSystemEvent(`Applying settings: URL=${apiInputUrl}, Profile=${selectedProfile}`, "API");')
app_code = app_code.replace('const data = await initializeEngine(selectedProfile, selectedDevice);', 'const data = await initializeEngine(selectedProfile, embedderModel, llmModel, embedderDevice, llmDevice, treesStorage);')

with open("frontend/src/App.jsx", "w", encoding="utf-8") as f:
    f.write(app_code)
