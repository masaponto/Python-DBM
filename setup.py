from distutils.core import setup

setup(
    name='svm_dbm',
    version='0.3',
    description='Decision boundary maiking',
    author='masaponto',
    author_email='masaponto@gmail.com',
    url='masaponto.github.io',
    install_requires=['scikit-learn>=0.18.1', 'numpy'],
    py_modules=['svm_dbm'],
    package_dir={'': 'src'}
)
