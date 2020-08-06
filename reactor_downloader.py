import os
import time
import argparse
import logging
import skimage.io
import numpy as np
from threading import Thread
from collections import defaultdict
MAX_RETRY = 3

max_range = 2000


def main():
	os.makedirs('icons', exist_ok=True)
	for i in range(1, max_range):
		file = str(i) + ".png"
		if not os.path.isfile(file):
			try:
				im = skimage.io.imread("http://img1.reactor.cc/pics/award/" + str(i))
				skimage.io.imsave("icons/%s.png" % str(i), im)
			except:
				print("Failed " + file)

main()
