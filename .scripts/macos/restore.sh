#!/bin/bash
set -e
set -x

pip install six

brew update
brew install cmake
brew install boost
brew install flatbuffers
