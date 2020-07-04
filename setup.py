import re
import sys

from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

extras = {
    'test': [
        'filelock',
        'pytest',
        'pytest-forked',
        'pandas'
        'maddpg'
        'multiagent'
    ],
    'mpi': [
        'mpi4py'
    ]
}

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]

extras['all'] = all_deps

setup(name='marl_formation',
      packages=[package for package in find_packages()
                if package.startswith('marl_formation')],
      install_requires=[
          'baselines>=0.1.5',
          'gym>=0.15.4, <0.16.0',
          'scipy',
          'tqdm',
          'joblib',
          'cloudpickle',
          'click',
          'opencv-python',
          'glog',
          'networkx',
          'pyyaml',
          'matplotlib'
      ],
      extras_require=extras,
      description='multi robot formation control based on marl',
      author='Xiangyu Liu',
      url='https://github.com/LXYYY/marl_formation',
      author_email='xiangyu002@e.ntu.edu.sg',
      version='0.0.1')

# ensure there is some tensorflow build with version above 1.4
import pkg_resources

tf_pkg = None
for tf_pkg_name in ['tensorflow', 'tensorflow-gpu', 'tf-nightly', 'tf-nightly-gpu']:
    try:
        tf_pkg = pkg_resources.get_distribution(tf_pkg_name)
    except pkg_resources.DistributionNotFound:
        pass
assert tf_pkg is not None, 'TensorFlow needed, of version above 1.4'
from distutils.version import LooseVersion

assert LooseVersion(re.sub(r'-?rc\d+$', '', tf_pkg.version)) >= LooseVersion('1.4.0')
