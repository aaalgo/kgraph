#!/bin/bash

vagrant up
vagrant ssh -c /vagrant/kgraph/make-release-vagrant.sh
#vagrant halt
version=`cat version`
release=kgraph-$version-x86_64
if [ ! -f $release.tar.gz ]
then
    echo FAILED
    exit
fi

#scp $release.tar.gz cloud:/var/www/kgraph/releases


