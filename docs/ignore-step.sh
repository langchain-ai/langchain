#!/bin/bash

echo "VERCEL_ENV: $VERCEL_ENV"
echo "VERCEL_GIT_COMMIT_REF: $VERCEL_GIT_COMMIT_REF"


if [ "$VERCEL_ENV" == "production" ] || [ "$VERCEL_GIT_COMMIT_REF" == "master" ] || [ "$VERCEL_GIT_COMMIT_REF" == "v0.1" ]; then 
    exit 1; 
else 
    git log -n 50 --pretty=format:"%s" -- . ../templates | grep -v '(#' && exit 1 || exit 0;
    # git log -n 50 --pretty=format:"%s" -- . ../templates | grep -v '(#'
    # if [ $? -eq 0 ]; then
    #     exit 1
    # else
    #     exit 0
    # fi
fi
