import numpy as np
import functools
from scipy.spatial import cKDTree
from src.percolation.connectivity import DefectConnect

class PlanarCapSim:
	def __init__(self,
				 dimx:float=30,
				 dimy:float=30,
				 dimz:float=10,
				 radius:float=0.45,
				 max_defects:int=10000,
				 rebuild_thresh:int=50,
				 seed:int=42):
		assert dimx > radius
		assert dimy > radius
		assert dimz > radius
		assert radius > 0.
		assert max_defects > 0
		assert rebuild_thresh > 0

		# dimensions and constants
		self.dimx = dimx
		self.dimy = dimy
		self.dimz = dimz
		self.radius = radius
		self.max_defects = max_defects
		self.rebuild_thresh = rebuild_thresh

		# generate random sample
		self.seed = seed
		self.rng = np.random.default_rng(seed=self.seed)

		self.reset()

	def reset(self) -> None:
		# KDtree
		self.tree = None
		self.tree_points = []
		self.new_points = []

		# defect connect
		self.defect_conn = DefectConnect()

		# results
		self.is_simulated = False
		self.defect_num = 0
		self.defect_volume_norm = None

	def _find_neighbors(self, p:np.ndarray, distance:float, workers:int=1) -> [int]:
		neighbors = []

		if self.tree is not None:
			neighbors.extend(self.tree.query_ball_point(x=p, r=distance, workers=workers))

		idx_offset = len(self.tree_points)
		for delta_idx, q in enumerate(self.new_points):
			if np.linalg.norm(p-q) < distance:
				neighbors.append(idx_offset + delta_idx)

		return neighbors

	def find_neighbors(self, p:np.ndarray, workers:int=1) -> [int]:
		return self._find_neighbors(p=p, distance=2*self.radius, workers=workers)

	def union_defects(self, p:np.ndarray, cur_idx:int, neighbors:[int]) -> None:
		self.defect_conn.add(x=cur_idx)
		for uidx in neighbors:
			self.defect_conn.union(x=uidx, y=cur_idx)

		if (self.dimz - p[2]) < self.radius:
			self.defect_conn.union_anode(x=cur_idx)

		if p[2] < self.radius:
			self.defect_conn.union_cathode(x=cur_idx)

	def build_tree(self) -> None:
		if len(self.new_points) > 0:
			self.tree_points.extend(self.new_points)
			self.new_points = []
			self.tree = cKDTree(self.tree_points)

	def insert_tree_point(self, p:np.ndarray) -> None:
		self.new_points.append(p)

		# rebuild KDTree
		if len(self.new_points) == self.rebuild_thresh:
			self.build_tree()

	def get_sample(self) -> np.ndarray:
		point = self.rng.uniform(low=[0., 0., 0.],
								 high=[self.dimx, self.dimy, self.dimz])
		return point

	def is_broken(self) -> bool:
		return self.defect_conn.is_connected()

	def simulate(self, vol_est_samples:int=500000, workers:int=1):
		self.reset()
		
		for cur_idx in range(self.max_defects):
			p = self.get_sample()

			neighbors = self.find_neighbors(p=p, workers=workers)
			self.union_defects(p=p, cur_idx=cur_idx, neighbors=neighbors)
			self.insert_tree_point(p=p)

			self.defect_num += 1

			if self.is_broken():
				break

		self.is_simulated = True
		self.build_tree()
		self.estimate_defect_volume(samples=vol_est_samples, workers=workers)

	def estimate_defect_volume(self, samples:int=500000, workers:int=1):
		assert self.is_simulated, "Error: need to run simulaiton first."
		assert samples > 0
		v_mc_pts = self.rng.uniform(low=[0., 0., 0.],
									high=[self.dimx, self.dimy, self.dimz],
									size=(samples, 3))

		dist, _ = self.tree.query(x=v_mc_pts, k=1, workers=workers)
		inside_samples = dist < self.radius
		self.defect_volume_norm = inside_samples.mean()

	def get_sim_results(self) -> (bool, int, float):
		assert self.is_simulated
		return self.is_broken(), self.defect_num, self.defect_volume_norm


def PlanarCapSim_simulation(seed:int,
							dimx:float=30,
							dimy:float=30,
							dimz:float=10,
							radius:float=0.45,
							max_defects:int=10000,
							rebuild_thresh:int=50,
							workers:int=1,
							vol_est_samples:int=500000)\
							-> (bool, int, float):
	sim = PlanarCapSim(dimx=dimx,
					   dimy=dimy,
					   dimz=dimz,
					   radius=radius,
					   max_defects=max_defects,
					   rebuild_thresh=rebuild_thresh,
					   seed=seed)
	sim.simulate(vol_est_samples=vol_est_samples,
				 workers=workers)
	return sim.get_sim_results()

def PlanarCapSim_create_wrapper(dimx:float=30,
								dimy:float=30,
								dimz:float=10,
								radius:float=0.45,
								max_defects:int=10000,
								rebuild_thresh:int=50,
								workers:int=1,
								vol_est_samples:int=500000)\
								-> functools.partial:
	return functools.partial(PlanarCapSim_simulation,
							 dimx=dimx,
							 dimy=dimy,
							 dimz=dimz,
							 radius=radius,
							 max_defects=max_defects,
							 rebuild_thresh=rebuild_thresh,
							 workers=workers,
							 vol_est_samples=vol_est_samples)



if __name__ == "__main__":
	pass