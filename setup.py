from distutils.core import setup

setup(
    name='svm_dbm',
    version='0.2',
    description='Decision boundary maiking',
    author='masaponto',
    author_email='masaponto@gmail.com',
    url='masaponto.github.io',
    install_requires=['scikit-learn==0.18.1',
                      'scipy==0.18.1', 'numpy==1.11.2'],
    py_modules=['svm_dbm'],
    package_dir={'': 'src'}
)
