import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Copy N Random Samples From Source To Target Directory")
parser.add_argument('-s', '--source', type=str, required=True)
parser.add_argument('-p', '--prefix', default="", type=str)
parser.add_argument('-f', '--files', nargs="+", type=str)
parser.add_argument('-d', '--destination', type=str, required=True)
args = parser.parse_args()

if not os.path.isdir(args.destination):
	os.system("mkdir -p '%s'" % args.destination)

for f in args.files:
	f = args.prefix + f
	os.system("cp '%s' '%s'" % (os.path.join(args.source, f), os.path.join(args.destination, f)))
