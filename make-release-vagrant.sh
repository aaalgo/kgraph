#!/bin/bash

rm -rf vagrant
git clone /vagrant
cd vagrant
make
make release
version=`cat version`
release=kgraph-$version-x86_64
mv kgraph-release $release
tar zcvf $release.tar.gz $release
cp $release.tar.gz /vagrant
