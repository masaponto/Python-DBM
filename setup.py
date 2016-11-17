from distutils.core import setup

setup(
    name='dbm',
    version='0.2',
    description='decision boundary maiking using mlp',
    author='masaponto',
    author_email='masaponto@gmail.com',
    url='masaponto.github.io',
    install_requires=['scikit-learn==0.18.1',
                      'scipy==0.18.1', 'numpy==1.11.2'],
    py_modules=["dbm"],
    package_dir={'': 'src'}
)
