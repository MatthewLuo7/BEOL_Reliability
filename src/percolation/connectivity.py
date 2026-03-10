class UnionFind:
	"""
	Union-Find algorithm to manage and record the nodes
	connectivity in graphs.
	Algorithm Credit: ChatGPT
	Comment Credict: Erjing Luo
	"""
	def __init__(self) -> None:
		"""
		Parent nodes of each nodes in the graphs.
		Each node has a parent node that is in the same graph with itself.
		This parent node can be itself and is updatable.
		If multiple nodes are in the same graph, we hope to assign the
		same parent node for them, as it is sufficient to represent the
		connectivity. The way to get parent node is achieved as a 'find()'
		function. It traces the root parent node in recursive fashion,
		while also dynamically updating the parent node as the root
		parent node to compress connectivity path.
		"""
		self.parent = {}

	def add(self, x:int) -> None:
		"""
		Add a new node

		Parameters
		----------
		x: int
			The index of the node to be inserted
		"""
		assert x not in self.parent, f"Error: Key '{x}' has already been in the dictionary"
		self.parent[x] = x

	def find(self, x:int) -> int:
		"""
		Find the root parent node recursively and update the
		parent nodes as the root parent node for path compression.

		Parameters
		----------
		x: int
			The index of the start node
		
		Returns
		-------
		int
			The index of the root parent node
		"""
		if self.parent[x] != x:
			self.parent[x] = self.find(self.parent[x])

		return self.parent[x]

	def union(self, x:int, y:int) -> None:
		"""
		Unionize the node x and node y into the same graph.
		This means their graphs need to be connected.

		Parameters
		----------
		x: int
			The index of node x
		y: int
			The index of node y
		"""
		rx = self.find(x)
		ry = self.find(y)

		if rx != ry:
			self.parent[ry] = rx

class DefectConnect(UnionFind):
	"""
	Manage defect connectivity problem as a union-find model
	Credit: Erjing Luo
	"""
	def __init__(self) -> None:
		"""
		Add anode and cathode
		"""
		super().__init__()
		self._ANODE = -1
		self._CATHODE = -2
		self.add(self._ANODE)		# Anode
		self.add(self._CATHODE)		# Cathode

	def union_anode(self, x:int) -> None:
		"""
		Unionize a node to anode

		Parameters
		----------
		x: int
			The index of the input node
		"""
		self.union(x=self._ANODE, y=x)

	def union_cathode(self, x:int) -> None:
		"""
		Unionize a node to cathode

		Parameters
		----------
		x: int
			The index of the input node
		"""
		self.union(x=self._CATHODE, y=x)

	def is_connected(self) -> bool:
		"""
		If anode and cathode are connected or not
		"""
		flag = (self.find(self._ANODE) == self.find(self._CATHODE))
		return flag

	def reset(self) -> None:
		"""
		Reset
		"""
		self.parent = {}

if __name__ == "__main__":
	pass