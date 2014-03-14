#!/bin/bash

vagrant up
vagrant ssh -c /vagrant/make-release-vagrant.sh
vagrant halt

