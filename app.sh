#!/bin/bash
docker build -t nba . && docker run --rm -p 5000:5000 nba
