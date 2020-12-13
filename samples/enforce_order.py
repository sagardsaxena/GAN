import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Copy N Random Samples From Source To Target Directory")
parser.add_argument('-s', '--source', type=str, required=True)
parser.add_argument('-f', '--files', nargs="+", type=str)
args = parser.parse_args()

for i, f in enumerate(args.files):
	g = "%d_%s" % (i, f)
	os.system("mv '%s' '%s'" % (os.path.join(args.source, f), os.path.join(args.source, g)))
