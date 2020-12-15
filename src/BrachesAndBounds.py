import numpy as np
import pandas as pd
import string


class Node:

	def __init__(self, frame: pd.DataFrame = None, weight: float = None, path: str = 'root', parent = None, null_matrix=None):
		self.is_leaf = True
		self.left = None
		self.right = None
		self.frame = frame
		self.parent = parent
		self.weight = weight
		self.path = path
		self.is_forgoten = False
		self.null_matrix = null_matrix

	def grow_left(self, frame: pd.DataFrame, weight: float, path: str,  null_matrix) -> None:
		self.left = Node(frame, weight, path, self, null_matrix)
		self._check_growth()

	def grow_right(self, frame: pd.DataFrame, weight: float, path: str) -> None:
		self.right = Node(frame, weight, path, self)
		self._check_growth()

	def _check_growth(self) -> None:
		if self.left is not None and self.right is not None:
			self.is_leaf = False

	def find_least_leaf(self):
		if self.is_leaf:

			return self
		elif self.left is not None and self.right is not None:
			left = self.left.find_least_leaf()
			right = self.right.find_least_leaf()
			if left.weight < right.weight:
				right.is_forgoten = True
				return left
			else:
				left.is_forgoten = True
				return right

	def backward(self):
		total_path = list()

		cursor = self
		while True:
			total_path.append(cursor.path)
			if cursor.parent is not None:
				cursor = cursor.parent
			else:
				return total_path

class BranchAndBoundSolver:

	def __init__(self, initial_matrix: np.ndarray):

		labels = list(string.ascii_uppercase[:initial_matrix.shape[0]])

		self.initial_frame = pd.DataFrame(
			initial_matrix.copy(),
			columns=labels,
			index=labels
		)

		self.root = Node(
			*BranchAndBoundSolver.reduction(self.initial_frame)
		)

	def solve(self):

		while self.root.find_least_leaf().frame.to_numpy().shape != (2, 2):
			leaf = self.root.find_least_leaf()

			max_element_value, max_element_position, pos_names, lf = BranchAndBoundSolver.max_element_analysis(
				leaf.frame)

			if leaf.is_forgoten:
				leaf.frame, _ = BranchAndBoundSolver.ban_element(leaf.frame, pos_names)
				max_element_value, max_element_position, pos_names, lf = BranchAndBoundSolver.max_element_analysis(
					leaf.frame)

			leaf.grow_left(*BranchAndBoundSolver.remove_path(leaf.frame, max_element_position, leaf.weight, pos_names), lf)
			leaf.null_matrix = lf.copy(deep=True)
			leaf.grow_right(leaf.frame.copy(), leaf.weight + max_element_value, f'skip: {pos_names}')

		last_leaf = self.root.find_least_leaf()
		last_frame = last_leaf.frame.copy(deep=True)

		pre, last = BranchAndBoundSolver.special_max_element_analysis(last_frame)

		self.path = list(reversed(last_leaf.backward()))
		self.path.append(f'add: {pre}')
		self.path.append(f'add: {last}')

		print(self.path)

	@staticmethod
	def remove_path(frame: pd.DataFrame, max_element_position, last_penalty, pos_names):

		local_frame = frame.copy(deep=True)
		i, j = max_element_position
		begin, end = pos_names

		try:
			local_frame.loc[begin][end] = np.nan
		except:
			pass

		try:
			local_frame.loc[end][begin] = np.nan
		except:
			pass

		local_matrix = local_frame.to_numpy().copy()

		local_matrix = np.delete(local_matrix, i, axis=0)
		local_matrix = np.delete(local_matrix, j, axis=1)

		columns = local_frame.columns[local_frame.columns != end]
		index = local_frame.index[local_frame.index != begin]

		new_frame = pd.DataFrame(local_matrix.copy(), columns=columns, index=index)

		reduced_new_frame, penalty = BranchAndBoundSolver.reduction(new_frame)

		return reduced_new_frame.copy(deep=True), penalty + last_penalty, f'add: {pos_names}'

	@staticmethod
	def special_max_element_analysis(frame: pd.DataFrame):
		local_matrix = frame.to_numpy().copy()
		new_matrix = np.zeros(shape=local_matrix.shape)

		for i in np.arange(local_matrix.shape[0]):
			for j in np.arange(local_matrix.shape[1]):
				stash = local_matrix[i, j]
				local_matrix[i, j] = np.nan
				axis0_min = np.nanmin(local_matrix[i])
				axis1_min = np.nanmin(local_matrix.T[j])
				new_matrix[i, j] = axis0_min + axis1_min
				local_matrix[i, j] = stash

		max_value = np.max(new_matrix)

		first_value_pos = np.argwhere(np.isnan(new_matrix)).T[0]
		second_value_pos = np.argwhere(np.isnan(new_matrix)).T[1]

		first_max_value_indecies = tuple(first_value_pos)
		second_max_value_indecies = tuple(second_value_pos)

		local_frame = pd.DataFrame(new_matrix, columns=frame.columns.copy(), index=frame.index.copy())

		first_begin = local_frame.index[first_max_value_indecies[0]]
		first_end = local_frame.columns[first_max_value_indecies[1]]

		second_begin = local_frame.index[second_max_value_indecies[0]]
		second_end = local_frame.columns[second_max_value_indecies[1]]

		first_pos_names = (first_begin, first_end)
		second_pos_names = (second_begin, second_end)

		return first_pos_names, second_pos_names

	@staticmethod
	def max_element_analysis(frame: pd.DataFrame):
		local_matrix = frame.to_numpy().copy()
		new_matrix = np.zeros(shape=local_matrix.shape)

		for i in np.arange(local_matrix.shape[0]):
			for j in np.arange(local_matrix.shape[1]):
				stash = local_matrix[i, j]
				local_matrix[i, j] = np.nan

				axis0_min = np.nanmin(local_matrix[i])
				axis1_min = np.nanmin(local_matrix.T[j])

				new_matrix[i, j] = axis0_min + axis1_min

				local_matrix[i, j] = stash

		max_value = np.nanmax(new_matrix)

		max_value_position = np.array(
			np.where(new_matrix == max_value)
		).T[0]

		max_value_indecies = tuple(max_value_position)

		local_frame = pd.DataFrame(new_matrix, columns=frame.columns.copy(), index=frame.index.copy())

		begin = local_frame.index[max_value_indecies[0]]
		end = local_frame.columns[max_value_indecies[1]]

		pos_names = (begin, end)

		return max_value, max_value_position, pos_names, local_frame.copy(deep=True)

	@staticmethod
	def reduction(frame: pd.DataFrame):
		local_matrix = frame.to_numpy().copy()
		weight = 0

		axis1_mins = np.nanmin(local_matrix, axis=1).reshape(-1, 1)
		weight += axis1_mins.sum()
		local_matrix -= axis1_mins

		axis0_mins = np.nanmin(local_matrix, axis=0)
		weight += axis0_mins.sum()
		local_matrix -= axis0_mins

		new_frame = pd.DataFrame(local_matrix, columns=frame.columns, index=frame.index)

		return new_frame, weight

	@staticmethod
	def ban_element(frame: pd.DataFrame, pos_names):
		local_frame = frame.copy(deep=True)
		local_frame.loc[pos_names[0]][pos_names[1]] = np.nan

		return BranchAndBoundSolver.reduction(local_frame)


def test1():
	A = np.array(
		[
			[np.nan, 68, 73, 24, 70, 9],
			[58, np.nan, 16, 44, 11, 92],
			[63, 9, np.nan, 86, 13, 18],
			[17, 34, 76, np.nan, 52, 70],
			[60, 18, 3, 45, np.nan, 58],
			[16, 82, 11, 60, 48, np.nan]
		],
		dtype=float
	)

	solver = BranchAndBoundSolver(A)
	solver.solve()


def test2():
	A = np.array(
		[
			[np.nan, 20, 18, 12, 8],
			[5, np.nan, 14, 7, 11],
			[12, 18, np.nan, 6, 11],
			[11, 17, 11, np.nan, 12],
			[5, 5, 5, 5, np.nan]
		],
		dtype=float
	)

	solver = BranchAndBoundSolver(A)
	solver.solve()


def test3():
	A = np.array(
		[
			[np.nan, 4, 7, 9, 4],
			[3, np.nan, 1, 3, 2],
			[10, 9, np.nan, 1, 7],
			[4, 6, 1, np.nan, 9],
			[5, 2, 3, 7, np.nan]
		],
		dtype=float
	)

	solver = BranchAndBoundSolver(A)
	solver.solve()

	return None


if __name__ == '__main__':
	test3()

	a = pd.DataFrame(
		np.array([[np.nan, 0], [0, np.nan]]),
		columns=['B', 'E'],
		index=['A', 'E']
	)

	pass

