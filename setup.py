from setuptools import setup, find_packages

setup(
  name = 'hkey-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Hierarchical Keys (HKey) of weights to exploit activation sparsity.',
  author = 'Ali Kore',
  author_email = 'akore654@gmail.com',
  long_description=open('README.md', 'r').read(),
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/a-kore/hkey-pytorch',
  keywords = [
    'artificial intelligence',
    'AI',
    'machine learning',
    'deep learning',
    'pytorch',
    'transformers',
    'hierarchical keys',
    'hkey',
    'activation sparsity',
  ],  
  install_requires=[
    'torch>=1.6',
    'transformers>=4.0',
    'fast-pytorch-kmeans>=0.1.9',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',
  ],
)