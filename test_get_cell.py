from src.lattice import LatticeOrchestrator
orch = LatticeOrchestrator("trees")
orch.load_from_database()
c = orch.loaded_cells.get('PANDAS_READ_CSV')
print("Cell:", c)
if c:
    print("Type name:", c.inputs.type_name)
    print("State:", c.inputs.state)
    print("Node type:", getattr(c, 'node_type', None))
    print("Is Micro:", c.type == 'micro')
