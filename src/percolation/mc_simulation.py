import numpy as np
from numpy.random import SeedSequence

from itertools import chain
import functools
import multiprocessing
from tqdm import tqdm

# def mc_simulate(sim_wrapper:functools.partial,
# 				sample_num:int=1000,
# 				randseq_entropy:int=42,
# 				chunk_size:int=8,
# 				cpu_num:int=-1)\
# 				-> (np.ndarray, np.ndarray, np.ndarray):
# 	ss = SeedSequence(randseq_entropy)
# 	child_seeds = ss.spawn(sample_num)

# 	cpu_num = multiprocessing.cpu_count() if cpu_num <= 0 else cpu_num

# 	try:
# 		pool = multiprocessing.Pool(cpu_num)
# 		results = pool.map_async(sim_wrapper, child_seeds, chunksize=chunk_size).get()
# 		pool.close()
# 		pool.join()

# 		# unzip results, as they have different data types (i.e. bool, int, float)
# 		break_list, defect_list, volume_list = zip(*results)
# 		return np.array(break_list), np.array(defect_list), np.array(volume_list)

# 	except KeyboardInterrupt:
# 		pool.terminate()
# 		pool.join()
# 		return


# def mc_simulate(sim_wrapper:functools.partial,
# 				sample_num:int=1000,
# 				randseq_entropy:int=42,
# 				chunk_size:int=8,
# 				cpu_num:int=-1)\
# 				-> (np.ndarray, np.ndarray, np.ndarray):

# 	ss = SeedSequence(randseq_entropy)
# 	child_seeds = ss.spawn(sample_num)

# 	cpu_num = multiprocessing.cpu_count() if cpu_num <= 0 else cpu_num

# 	try:
# 		with multiprocessing.Pool(cpu_num) as pool:

# 			results = list(
# 				tqdm(
# 					pool.imap(sim_wrapper, child_seeds, chunk_size),
# 					total=sample_num
# 				)
# 			)

# 		break_list, defect_list, volume_list = zip(*results)

# 		return np.array(break_list).astype(bool), np.array(defect_list).astype(np.int32), np.array(volume_list).astype(np.float32)

# 	except KeyboardInterrupt:
# 		pool.terminate()
# 		pool.join()
# 		return


def mc_simulate(sim_wrapper:functools.partial,
				sample_num:int=1000,
				randseq_entropy:int=42,
				chunk_size:int=8,
				cpu_num:int=-1)\
				-> (np.ndarray, np.ndarray, np.ndarray):

	ss = SeedSequence(randseq_entropy)
	child_seeds = ss.spawn(sample_num)

	cpu_num = multiprocessing.cpu_count() if cpu_num <= 0 else cpu_num

	try:
		with multiprocessing.Pool(cpu_num) as pool:

			results = list(
				tqdm(
					pool.imap(sim_wrapper, child_seeds, chunk_size),
					total=sample_num
				)
			)

		break_list, defect_list, p_list_list = zip(*results)

		return np.array(break_list).astype(bool), np.array(defect_list).astype(np.int32), np.array(list(chain.from_iterable(p_list_list))).astype(np.float32)

	except KeyboardInterrupt:
		pool.terminate()
		pool.join()
		return


if __name__ == "__main__":
	pass