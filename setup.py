from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="rs_fast_one_hot",
    version="0.1.0",
    rust_extensions=[RustExtension("rs_fast_one_hot.rs_fast_one_hot", binding=Binding.PyO3)],
    packages=["rs_fast_one_hot"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
    install_requires=['scikit-learn']
)