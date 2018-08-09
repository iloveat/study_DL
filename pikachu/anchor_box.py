# -*- coding: utf-8 -*-
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior
import matplotlib.pyplot as plt


def box_to_rect(box, color, line_width=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        fill=False, edgecolor=color, linewidth=line_width)


# shape: batch x channel x height x weight
n = 40
x = nd.random.uniform(shape=(1, 3, n, n))
y = MultiBoxPrior(x, sizes=[.5, .25, .1], ratios=[1, 2, .5])

boxes = y.reshape((n, n, -1, 4))
print boxes.shape

# The first anchor box centered on (20, 20)
# its format is (x_min, y_min, x_max, y_max)
print boxes[20, 20, 0, :]
anchors = boxes[20, 20, :, :]

colors = ['blue', 'green', 'red', 'black', 'magenta']

# size: 20x20,10x10,4x4,20sqrt(2)x20/sqrt(2),20sqrt(0.5)x20/sqrt(0.5)
plt.imshow(nd.ones((n, n, 3)).asnumpy())
for i in range(anchors.shape[0]):
    plt.gca().add_patch(box_to_rect(anchors[i, :]*n, colors[i]))
plt.show()





