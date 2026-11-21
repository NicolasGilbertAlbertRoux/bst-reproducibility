'An integral dynamic of seven sites, two mass tasks in memory at most simultaneously.'
import argparse
from concurrent.futures import ThreadPoolExecutor
from grid import run
import relax_nodes as nodal

def bounded_pool(max_workers=None, **kwargs):
    return ThreadPoolExecutor(max_workers=min(2,max_workers or 2), **kwargs)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('nodes',type=int)
    parser.add_argument('--dt',type=float,default=.09)
    args=parser.parse_args()
    nodal.ThreadPoolExecutor=bounded_pool
    run(args.nodes,'impedance',args.dt,.36)
