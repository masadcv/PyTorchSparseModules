import setuptools

def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(name='torchsparsemodules',
      version='0.0.2',
      description='Learnable sparse modules using PyTorch',
      long_description=readme(),
      long_description_content_type="text/markdown",
      classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
      ],
      keywords='learnable sparse modules pytorch',
      url='https://github.com/masadcv/PyTorchSparseModules',
      author='Muhammad Asad',
      author_email='muhammad.asad@qmul.ac.uk',
      license='BSD-3-Clause',
      packages=['torchsparsemodules'],
      install_requires=[
          'torch',
      ],
      zip_safe=False)
