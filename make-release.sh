#!/bin/bash

vagrant up
vagrant ssh -c /vagrant/make-release-vagrant.sh
vagrant halt
if [ ! -f kgraph-1.1-x86_64.tar.gz ]
then
    echo FAILED
    exit
fi

#scp kgraph-1.1-x86_64.tar.gz cloud:/var/www/kgraph/releases


