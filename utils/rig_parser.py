#-------------------------------------------------------------------------------
# Name:        rig_parser.py
# Purpose:     classes for skeleton and rig
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import numpy as np
from utils.tree_utils import TreeNode
try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q


class Info:
    """
    Wrap class for rig information
    """
    def __init__(self, filename=None):
        self.joint_pos = {}
        self.joint_skin = []
        self.root = None
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        with open(filename, 'r') as f_txt:
            lines = f_txt.readlines()
        for line in lines:
            word = line.split()
            if word[0] == 'joints':
                self.joint_pos[word[1]] = [float(word[2]), float(word[3]), float(word[4])]
            elif word[0] == 'root':
                root_pos = self.joint_pos[word[1]]
                self.root = TreeNode(word[1], (root_pos[0], root_pos[1], root_pos[2]))
            elif word[0] == 'skin':
                skin_item = word[1:]
                self.joint_skin.append(skin_item)
        self.loadHierarchy_recur(self.root, lines, self.joint_pos)

    def loadHierarchy_recur(self, node, lines, joint_pos):
        for li in lines:
            if li.split()[0] == 'hier' and li.split()[1] == node.name:
                pos = joint_pos[li.split()[2]]
                ch_node = TreeNode(li.split()[2], tuple(pos))
                node.children.append(ch_node)
                ch_node.parent = node
                self.loadHierarchy_recur(ch_node, lines, joint_pos)

    def save(self, filename):
        with open(filename, 'w') as file_info:
            for key, val in self.joint_pos.items():
                file_info.write(
                    'joints {0} {1:.8f} {2:.8f} {3:.8f}\n'.format(key, val[0], val[1], val[2]))
            file_info.write('root {}\n'.format(self.root.name))

            for skw in self.joint_skin:
                cur_line = 'skin {0} '.format(skw[0])
                for cur_j in range(1, len(skw), 2):
                    cur_line += '{0} {1:.4f} '.format(skw[cur_j], float(skw[cur_j+1]))
                cur_line += '\n'
                file_info.write(cur_line)

            this_level = self.root.children
            while this_level:
                next_level = []
                for p_node in this_level:
                    file_info.write('hier {0} {1}\n'.format(p_node.parent.name, p_node.name))
                    next_level += p_node.children
                this_level = next_level

    def save_as_skel_format(self, filename):
        fout = open(filename, 'w')
        this_level = [self.root]
        hier_level = 1
        while this_level:
            next_level = []
            for p_node in this_level:
                pos = p_node.pos
                parent = p_node.parent.name if p_node.parent is not None else 'None'
                line = '{0} {1} {2:8f} {3:8f} {4:8f} {5}\n'.format(hier_level, p_node.name, pos[0], pos[1], pos[2],
                                                                   parent)
                fout.write(line)
                for c_node in p_node.children:
                    next_level.append(c_node)
            this_level = next_level
            hier_level += 1
        fout.close()

    def normalize(self, scale, trans):
        for k, v in self.joint_pos.items():
            self.joint_pos[k] /= scale
            self.joint_pos[k] -= trans


        this_level = [self.root]
        while this_level:
            next_level = []
            for node in this_level:
                node.pos /= scale
                node.pos = (node.pos[0] - trans[0], node.pos[1] - trans[1], node.pos[2] - trans[2])
                for ch in node.children:
                    next_level.append(ch)
            this_level = next_level

    def get_joint_dict(self):
        joint_dict = {}
        this_level = [self.root]
        while this_level:
            next_level = []
            for node in this_level:
                joint_dict[node.name] = node.pos
                next_level += node.children
            this_level = next_level
        return joint_dict

    def adjacent_matrix(self):
        joint_pos = self.get_joint_dict()
        joint_name_list = list(joint_pos.keys())
        num_joint = len(joint_pos)
        adj_matrix = np.zeros((num_joint, num_joint))
        this_level = [self.root]
        while this_level:
            next_level = []
            for p_node in this_level:
                for c_node in p_node.children:
                    index_parent = joint_name_list.index(p_node.name)
                    index_children = joint_name_list.index(c_node.name)
                    adj_matrix[index_parent, index_children] = 1.
                next_level += p_node.children
            this_level = next_level
        adj_matrix = adj_matrix + adj_matrix.transpose()
        return adj_matrix


class Skel:
    """
    Wrap class for skeleton topology
    """
    def __init__(self, filename=None):
        self.root = None
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        with open(filename, 'r') as fin:
            lines = fin.readlines()
        for li in lines:
            words = li.split()
            if words[5] == "None":
                self.root = TreeNode(words[1], (float(words[2]), float(words[3]), float(words[4])))
                if len(words) == 7:
                    has_order = True
                    self.root.order = int(words[6])
                else:
                    has_order = False
                break
        self.loadSkel_recur(self.root, lines, has_order)

    def loadSkel_recur(self, node, lines, has_order):
        if has_order:
            ch_queue = Q.PriorityQueue()
            for li in lines:
                words = li.split()
                if words[5] == node.name:
                    ch_queue.put((int(li.split()[6]), li))
            while not ch_queue.empty():
                item = ch_queue.get()
                li = item[1]
                ch_node = TreeNode(li.split()[1], (float(li.split()[2]), float(li.split()[3]), float(li.split()[4])))
                ch_node.order = int(li.split()[6])
                node.children.append(ch_node)
                ch_node.parent = node
                self.loadSkel_recur(ch_node, lines, has_order)
        else:
            for li in lines:
                words = li.split()
                if words[5] == node.name:
                    ch_node = TreeNode(words[1], (float(words[2]), float(words[3]), float(words[4])))
                    node.children.append(ch_node)
                    ch_node.parent = node
                    self.loadSkel_recur(ch_node, lines, has_order)

    def save(self, filename):
        fout = open(filename, 'w')
        this_level = [self.root]
        hier_level = 1
        while this_level:
            next_level = []
            for p_node in this_level:
                pos = p_node.pos
                parent = p_node.parent.name if p_node.parent is not None else 'None'
                line = '{0} {1} {2:8f} {3:8f} {4:8f} {5}\n'.format(hier_level, p_node.name, pos[0], pos[1], pos[2], parent)
                fout.write(line)
                for c_node in p_node.children:
                    next_level.append(c_node)
            this_level = next_level
            hier_level += 1
        fout.close()

    def normalize(self, scale, trans):
        this_level = [self.root]
        while this_level:
            next_level = []
            for node in this_level:
                node.pos /= scale
                node.pos = (node.pos[0] - trans[0], node.pos[1] - trans[1], node.pos[2] - trans[2])
                for ch in node.children:
                    next_level.append(ch)
            this_level = next_level

    def get_joint_pos(self):
        joint_pos = {}
        this_level = [self.root]
        while this_level:
            next_level = []
            for node in this_level:
                joint_pos[node.name] = node.pos
                next_level += node.children
            this_level = next_level
        return joint_pos

    def adjacent_matrix(self):
        joint_pos = self.get_joint_pos()
        joint_name_list = list(joint_pos.keys())
        num_joint = len(joint_pos)
        adj_matrix = np.zeros((num_joint, num_joint))
        this_level = [self.root]
        while this_level:
            next_level = []
            for p_node in this_level:
                for c_node in p_node.children:
                    index_parent = joint_name_list.index(p_node.name)
                    index_children = joint_name_list.index(c_node.name)
                    adj_matrix[index_parent, index_children] = 1.
                next_level += p_node.children
            this_level = next_level
        adj_matrix = adj_matrix + adj_matrix.transpose()
        return adj_matrix
