from distutils.core import setup

setup(
    name='dbm_mlp',
    version='0.1',
    description='decision boundary maiking using mlp',
    author='masaponto',
    author_email='masaponto@gmail.com',
    url='masaponto.github.io',
    install_requires=['scikit-learn==0.17.1', 'scipy==0.17.0', 'numpy==1.10.4'],
    py_modules = ["dbm_mlp", "three_layer_mlp"],
    package_dir = {'': 'src'}
)
