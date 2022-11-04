#!/bin/bash
: << 'COMMENT'
This script is meant to run multiple DKNN trial configurations, based on 
whatever diretory is passed in for processing
COMMENT

# assume first argument is the search directory
for entry in "$1"/*
do
  if [[ $entry =~ trial-.*\.json$ ]];
  then
    python3 main.py $entry
  fi
done

