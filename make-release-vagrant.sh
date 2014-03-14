#!/bin/bash

rm -rf vagrant
git clone /vagrant
cd vagrant
make
make release
cp *.tar.gz /vagrant


