#!/bin/bash

vagrant up
vagrant ssh /vagrant/make-release-vagrant.sh
vagrant halt

