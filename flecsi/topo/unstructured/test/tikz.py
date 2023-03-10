import argparse
import json

parser = argparse.ArgumentParser(
    prog="tikz", description="create latex file describing the mesh"
)
parser.add_argument("mesh_file")
parser.add_argument(
    "process_color_files",
    metavar="file",
    nargs="+",
    help="files describing each process color",
)
args = parser.parse_args()


with open(args.mesh_file) as f:
    nvertices, ncells = map(int, f.readline().strip().split())
    verts = []
    cells = []

    for v in range(nvertices):
        x, y = map(float, f.readline().strip().split())
        verts.append((x, y))

    for c in range(ncells):
        cells.append(list(map(int, f.readline().strip().split())))

pcs = []

for pcfile in args.process_color_files:
    with open(pcfile) as f:
        pcs.append(json.loads(f.read()))


def cell_color(cid):
    for pc in pcs:
        if cid in pc["exclusive_cells_global"] or cid in pc["shared_cells_global"]:
            return pc["color"]
    return -1


header = """
\\documentclass[tikz, border=1mm]{standalone}
\\begin{document}
\\begin{tikzpicture}[]
"""


footer = """
\\end{tikzpicture}
\\end{document}
"""

cpalette = ["red", "blue", "green!50!black", "yellow!50!black", "gray"]

print(header)

for vid, v in enumerate(verts):
    print("\\coordinate (v{}) at ({}, {});".format(vid, v[0] * 10, v[1] * 10))

for cid, c in enumerate(cells):
    ccolor = cpalette[cell_color(cid)]
    lines = " -- ".join(["(v{})".format(v) for v in c])
    lines += " -- (v{}) node[above right,text={}] {{${}$}}".format(c[0], "white", cid)
    print("\\path[draw=black,fill={}] {};".format(ccolor, lines))

for vid, v in enumerate(verts):
    print("\\fill[] (v{}) circle (2pt) node[above]{{\\tiny ${}$}};".format(vid, vid))

print(footer)
