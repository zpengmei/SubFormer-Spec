from distutils.core import setup

setup(
    name='SubFormer-Spec',
    version='0.01',
    packages=['torch', 'torch_geometric', 'torch_scatter',
              'ogb', 'tqdm', 'networkx', 'rdkit', 'numpy', 'pandas',
              'scipy','scikit-learn'],
    url='',
    license='',
    author='Zihan Pengmei',
    author_email='zpengmei@uchicago.edu',
    description='Implementation of SubFormer with the Spectral token for molecular property prediction.'
)
