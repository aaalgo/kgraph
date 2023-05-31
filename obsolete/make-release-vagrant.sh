#!/bin/bash

rm -rf kgraph 
git clone /vagrant/kgraph
cd kgraph
make
make release
version=`cat version`
release=kgraph-$version-x86_64
rm -rf $release
mv kgraph-release $release
tar zcvf $release.tar.gz $release
cp $release.tar.gz /vagrant/kgraph
