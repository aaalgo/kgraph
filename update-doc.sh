#!/bin/bash

rm -rf gh-pages
mkdir gh-pages
pushd gh-pages
git clone -b gh-pages https://github.com/aaalgo/kgraph
popd
doxygen
pushd gh-pages/kgraph
git push
