from setuptools import setup, Extension

# Define the C extension module
pytfmbs_module = Extension(
    'pytfmbs',
    sources=['core.c'],
    include_dirs=['../../include'], # Path to tfmbs.h
    extra_compile_args=['-O2']
)

setup(
    name='pytfmbs',
    version='0.1',
    description='Python bindings for the Ternary Fabric Memory & Bus Specification',
    ext_modules=[pytfmbs_module]
)