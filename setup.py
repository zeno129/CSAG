from distutils.core import setup
setup(name='graph_sampling',
      version='0.1',
      package_dir={'graph_sampling': ''},
      packages=['graph_sampling.kronecker', 'graph_sampling.graph'],  # add rest here
      )
