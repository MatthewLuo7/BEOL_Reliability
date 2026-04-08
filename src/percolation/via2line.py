import numpy as np
import functools
from src.percolation.planar_capacitor import PlanarCapSim

def get_closest_cuboid_distance(p:np.ndarray, cuboid_dims:[float]) -> float:
	xmin, xmax, ymin, ymax, zmin, zmax = cuboid_dims
	x_closest = max(xmin, min(p[0], xmax))
	y_closest = max(ymin, min(p[1], ymax))
	z_closest = max(zmin, min(p[2], zmax))

	dx = p[0] - x_closest
	dy = p[1] - y_closest
	dz = p[2] - z_closest

	return (dx**2 + dy**2 + dz**2)**0.5

class Via2LineSim(PlanarCapSim):
	def __init__(self,
				 vm_offset:float,			# via misalignment, along the x dim
				 ll_space:float=10.5,		# line-line spacing, along the x dim
				 via_dim_x:float=10.5,		# via size (x dim)
				 via_dim_y:float=10.5,		# via size (y dim)
				 via_dim_z:float=10.5*2,	# via size (z dim)
				 # line_dim_x:float=10.5,	# line size (x dim),	equal to via size (x dim)
				 line_dim_y:float=10.5*2,	# line size (y dim)
				 line_dim_z:float=10.5*2, 	# line size (z dim)
				 radius:float=0.45,
				 max_defects:int=10000,
				 rebuild_thresh:int=50,
				 seed:int=42) -> None:

		assert vm_offset >= 0.
		assert ll_space > 0.
		assert via_dim_x > 0.
		assert via_dim_y > 0.
		assert via_dim_z > 0.
		assert line_dim_y > 0.
		assert line_dim_z > 0.
		assert radius > 0.
		assert max_defects > 0
		assert rebuild_thresh > 0

		assert vm_offset < min(via_dim_x, ll_space)
		assert via_dim_y <= line_dim_y

		self.vm_offset = vm_offset
		self.ll_space = ll_space
		self.via_dim_x = via_dim_x
		self.via_dim_y = via_dim_y
		self.via_dim_z = via_dim_z
		self.line_dim_y = line_dim_y
		self.line_dim_z = line_dim_z

		# ----- dims of the first line -----
		# x_min, x_max, y_min, y_max, z_min, z_max 
		self.line_a_dims = [0.,
							0. + self.via_dim_x,
							0.,
							0. + self.line_dim_y,
							0.,
							0. + self.line_dim_z
							]

		# ----- dims of the second line -----
		self.line_b_dims = [self.via_dim_x + self.ll_space,
							self.via_dim_x + self.ll_space + self.via_dim_x,
							0.,
							0. + self.line_dim_y,
							0.,
							0. + self.line_dim_z]

		# ----- dims of the via -----
		self.via_dims = [self.vm_offset,
						 self.vm_offset + self.via_dim_x,
						(self.line_dim_y - self.via_dim_y)/2,
						(self.line_dim_y - self.via_dim_y)/2 + self.via_dim_y,
						 self.line_dim_z,
						 self.line_dim_z + self.via_dim_z]

		self.radius = radius
		self.max_defects = max_defects
		self.rebuild_thresh = rebuild_thresh

		# generate random sample
		self.seed = seed
		self.rng = np.random.default_rng(seed=self.seed)

		self.reset()


	def union_defects(self, p:np.ndarray, cur_idx:int, neighbors:[int]) -> None:
		self.defect_conn.add(x=cur_idx)
		for uidx in neighbors:
			self.defect_conn.union(x=uidx, y=cur_idx)

		if (get_closest_cuboid_distance(p, self.line_a_dims) < self.radius) or\
		   (get_closest_cuboid_distance(p, self.via_dims) < self.radius):
			self.defect_conn.union_anode(x=cur_idx)

		if get_closest_cuboid_distance(p, self.line_b_dims) < self.radius:
			self.defect_conn.union_cathode(x=cur_idx)

	def get_sample(self, max_try:int=1000) -> np.ndarray:
		flag = False
		for _ in range(max_try):
			point = self.rng.uniform(low=[0., 0., 0.],
					high=[self.via_dim_x*2+self.ll_space, self.line_dim_y, self.line_dim_z])

			# if find a valid point (not within the cuboids)
			if (get_closest_cuboid_distance(point, self.line_a_dims) > 0.) and\
			   (get_closest_cuboid_distance(point, self.via_dims) > 0.) and\
			   (get_closest_cuboid_distance(point, self.line_b_dims) > 0.):
			   flag = True
			   break

		if flag:
			return point
		else:
			return None


def Via2LineSim_simulation(seed:int,
						   vm_offset:float,				# via misalignment, along the x dim
						   ll_space:float=10.5,			# line-line spacing, along the x dim
						   via_dim_x:float=10.5,		# via size (x dim)
						   via_dim_y:float=10.5,		# via size (y dim)
						   via_dim_z:float=10.5*2,		# via size (z dim)
						   line_dim_y:float=10.5*2,		# line size (y dim)
						   line_dim_z:float=10.5*2, 	# line size (z dim)
						   radius:float=0.45,
						   max_defects:int=10000,
						   rebuild_thresh:int=50,
						   workers:int=1)\
							-> (bool, int, [float]):
	sim = Via2LineSim(vm_offset=vm_offset,
					  ll_space=ll_space,
					  via_dim_x=via_dim_x,	
					  via_dim_y=via_dim_y,	
					  via_dim_z=via_dim_z,
					  line_dim_y=line_dim_y,
					  line_dim_z=line_dim_z,
					  radius=radius,
					  max_defects=max_defects,
					  rebuild_thresh=rebuild_thresh,
					  seed=seed)
	sim.simulate(workers=workers)
	return sim.get_sim_results()

def Via2LineSim_create_wrapper(vm_offset:float,				# via misalignment, along the x dim
							   ll_space:float=10.5,			# line-line spacing, along the x dim
							   via_dim_x:float=10.5,		# via size (x dim)
							   via_dim_y:float=10.5,		# via size (y dim)
							   via_dim_z:float=10.5*2,		# via size (z dim)
							   line_dim_y:float=10.5*2,		# line size (y dim)
							   line_dim_z:float=10.5*2, 	# line size (z dim)
							   radius:float=0.45,
							   max_defects:int=10000,
							   rebuild_thresh:int=50,
							   workers:int=1) -> functools.partial:
	return functools.partial(Via2LineSim_simulation,
							 vm_offset=vm_offset,
							 ll_space=ll_space,
							 via_dim_x=via_dim_x,	
							 via_dim_y=via_dim_y,	
							 via_dim_z=via_dim_z,
							 line_dim_y=line_dim_y,
							 line_dim_z=line_dim_z,
							 radius=radius,
							 max_defects=max_defects,
							 rebuild_thresh=rebuild_thresh,
							 workers=workers)