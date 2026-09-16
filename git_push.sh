#!/bin/bash

MESSAGE="${1:-update}"

git add .
git commit -m "$MESSAGE"
git push
