#!/bin/bash
pip3 uninstall tensorflow-rocm tensorflow -y
pip3 install tf-keras --no-deps
pip3 install https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.4/tensorflow_rocm-2.17.0-cp310-cp310-manylinux_2_28_x86_64.whl


