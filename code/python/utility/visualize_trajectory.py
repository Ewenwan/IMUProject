from OpenGL.GL import *
from OpenGL.GLUT import *

if __name__ == '__main__':
    import argparse
    import pandas

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser