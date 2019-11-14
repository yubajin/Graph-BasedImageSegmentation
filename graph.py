
from PIL import Image, ImageFilter
import numpy as np


'''
定义顶点(node)类，把每个像素定义为节点(顶点)。顶点有三个性质：
(1) parent，该顶点对应分割区域的母顶点，可以认为的分割区域的编号或者索引。
后面初始化时，把图像每个像素当成一个分割区域，所以每个像素的母顶点就是他们本身。
(2) rank，母顶点的优先级（每个顶点初始化为0），用来两个区域合并时，确定唯一的母顶点。
(3) size（每个顶点初始化为1），表示每个顶点作为母顶点时，所在分割区域的顶点数量。
当它被其他区域合并，不再是母顶点时，它的size不再改变。
'''
class Node:
    def __init__(self, parent, rank=0, size=1):
		#同一个区域，表示为同一个parent，具体可以表示为如果两区域相同/两像素点相同，color[parent]相同，即取一样的颜色
        self.parent = parent
        self.rank = rank
        self.size = size

    def __repr__(self):
        return '(parent=%s, rank=%s, size=%s)' % (self.parent, self.rank, self.size)

'''
(1) self.nodes初始化forest类的所有顶点列表，初始化时把图像每个像素作为顶点，
当成一个分割区域，每个像素的母顶点就是他们本身。forest.num_sets表示该图像当前的分割区域数量。
(2) size_of()，获取某个顶点的size，一般用来获得某个母顶点的size，即为母顶点所在分割区域的顶点数量。
(3) find()，获得该顶点所在区域的母顶点编号(索引)
'''
class Forest:
    def __init__(self, num_nodes):
        self.nodes = [Node(i) for i in range(num_nodes)]
        self.num_sets = num_nodes

	#该顶点所在区域的size(包含像素点个数)
    def size_of(self, i):
        return self.nodes[i].size

	#该顶点对应分割区域的母顶点
    def find(self, n):
        temp = n
        while temp != self.nodes[temp].parent:
            temp = self.nodes[temp].parent

        self.nodes[n].parent = temp
        return temp

	#合并两个区域
    def merge(self, a, b):
        if self.nodes[a].rank > self.nodes[b].rank:
            self.nodes[b].parent = a
            self.nodes[a].size = self.nodes[a].size + self.nodes[b].size
        else:
            self.nodes[a].parent = b
            self.nodes[b].size = self.nodes[b].size + self.nodes[a].size

            if self.nodes[a].rank == self.nodes[b].rank:
                self.nodes[b].rank = self.nodes[b].rank + 1

        self.num_sets = self.num_sets - 1

    def print_nodes(self):
        for node in self.nodes:
            print(node)

# 创建边，方向由(x,y)指向(x1,y1)，大小为梯度值
def create_edge(img, width, x, y, x1, y1, diff):
    #按照我们阅读的顺序，从上到下读每一行，每一行从左到右的顺序编号
    vertex_id = lambda x, y: y * width + x
    w = diff(img, x, y, x1, y1)
    return (vertex_id(x, y), vertex_id(x1, y1), w)

#创建图，对每个顶点，←↑↖↗创建四条边，达到8-邻域的效果，自此完成图的构建。
def build_graph(img, width, height, diff, neighborhood_8=False):
    graph_edges = []
    for y in range(height):
        for x in range(width):
            if x > 0:
                graph_edges.append(create_edge(img, width, x, y, x-1, y, diff))

            if y > 0:
                graph_edges.append(create_edge(img, width, x, y, x, y-1, diff))

            if neighborhood_8:
                if x > 0 and y > 0:
                    graph_edges.append(create_edge(img, width, x, y, x-1, y-1, diff))

                if x > 0 and y < height-1:
                    graph_edges.append(create_edge(img, width, x, y, x-1, y+1, diff))

    return graph_edges

'''
初次分割后的图像，对于其中定点数均小于min_size的两个相邻区域，进行合并。
'''
def remove_small_components(forest, graph, min_size):
    for edge in graph:
        a = forest.find(edge[0])
        b = forest.find(edge[1])

        if a != b and (forest.size_of(a) < min_size or forest.size_of(b) < min_size):
            forest.merge(a, b)

    return  forest


'''
(1) 首先初始化forest
(2) 对所有边，根据其权值从小到大排序
(3) 初始化区域内部差列表
(4) 从小到大遍历所有边，如果顶点在两个区域，且权值小于两个顶点所在区域的内部差(threshold[])，
则合并这两个区域，找到合并后区域的新母顶点，更新该顶点对应的区域内部差（threshold[]）:
threshold[i]=Int(Ci)+ k/|Ci|
Int(Ci)为顶点i所在区域的内部差；∣Ci∣为该区域的顶点数量；
k为可调参数，k过大，导致更新时区域内部差过大，导致过多的区域进行合并，最终造成图像分割粗糙，反之，k过小，容易导致图像分割太精细。
因为遍历时是从小到大遍历，所以如果合并，这条边的权值一定是新区域所有边最大的权值，
即为该新区域的内部差，因此Int(Ci)=weight(edge)
'''
def segment_graph(graph_edges, num_nodes, const, min_size, threshold_func):
    # Step 1: initialization
    forest = Forest(num_nodes)
    weight = lambda edge: edge[2]#function(edge)=>return edge[2]
    sorted_graph = sorted(graph_edges, key=weight)#将构建好的图按照节点间的不相似度从小到大排序
    threshold = [ threshold_func(1, const) for _ in range(num_nodes) ]#每个区域的阈值，初始化时为每个像素点赋值阈值；threshold_func(1, const)==>(const*1.0 / 1)

    # Step 2: merging
    for edge in sorted_graph:
        parent_a = forest.find(edge[0])
        parent_b = forest.find(edge[1])

		#类间差距小于类内差距时，condition为True，表明可以合并
        a_condition = weight(edge) <= threshold[parent_a]
        b_condition = weight(edge) <= threshold[parent_b]

		#如果顶点在两个区域，且权值小于两个顶点所在区域的内部差(threshold[])，则合并这两个区域
        if parent_a != parent_b and a_condition and b_condition:
            forest.merge(parent_a, parent_b)
            a = forest.find(parent_a)
            threshold[a] = weight(edge) + threshold_func(forest.nodes[a].size, const)

    return remove_small_components(forest, sorted_graph, min_size)

#计算两像素点的差距
def diff(img, x1, y1, x2, y2):
    _out = np.sum((img[x1, y1] - img[x2, y2]) ** 2)
    return np.sqrt(_out)

#计算T(t)
def threshold(size, const):
    return (const * 1.0 / size)

if __name__ == '__main__':
	image_file = Image.open('assets/yu.jpg')

	sigma = 0
	neighbor = 8
	K = 3000
	min_comp_size = 20

	size = image_file.size  # (width, height) in Pillow/PIL

	# Gaussian Filter
	smooth = image_file.filter(ImageFilter.GaussianBlur(sigma))
	smooth = np.array(smooth)

	graph_edges = build_graph(smooth, size[1], size[0], diff, neighbor == 8)
	forest = segment_graph(graph_edges, size[0] * size[1], K, min_comp_size, threshold)