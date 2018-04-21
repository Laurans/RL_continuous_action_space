import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_level = int(np.ceil(np.log(capacity+1))+1)
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(self.capacity, dtype=np.int32)
        self.size = 0
        self.cursor = 0

    def add(self, contents, value):
        index = self.cursor
        self.data[index] = contents
        self.cursor = (self.cursor+1) % self.capacity
        self.size = min(self.size+1, self.capacity)


        self.val_update(index, value)

    def val_update(self, index, value):
        tree_index = 2**(self.tree_level-1)-1+index
        diff = value-self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex-1)//2)
            self.reconstruct(tindex, diff)

    def get_val(self, index):
        tree_index = 2**(self.tree_level-1)-1+index
        return self.tree[tree_index]

    def total(self):
        return self.tree[0]

    def find(self, value, norm=True):
        if norm:
            value *= self.tree[0]
        return self._find(value, 0)

    def _find(self, value, index):
        if 2**(self.tree_level-1)-1 <= index:
            data_index = index-(2**(self.tree_level-1)-1)
            return (self.data[data_index], self.tree[index], data_index)

        left = self.tree[2*index+1]
        if value <= left:
            return self._find(value, 2*index+1)
        else:
            return self._find(value-left, 2*(index+1))

    def print_tree(self):
        for k in range(1, self.tree_level + 1):
            for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
                print(self.tree[j], end=' | ')
            print()
