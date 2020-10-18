import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--turn', type=int)
args=parser.parse_args()
print(type(args.turn))
