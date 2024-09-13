#!/bin/bash

echo "VERCEL_ENV: $VERCEL_ENV"
echo "VERCEL_GIT_COMMIT_REF: $VERCEL_GIT_COMMIT_REF"


if [ "$VERCEL_ENV" == "production" ] || \
    [ "$VERCEL_GIT_COMMIT_REF" == "master" ] || \
    [ "$VERCEL_GIT_COMMIT_REF" == "v0.1" ] || \
    [ "$VERCEL_GIT_COMMIT_REF" == "v0.2" ] || \
    [ "$VERCEL_GIT_COMMIT_REF" == "v0.3rc" ]
then 
     echo "✅ Production build - proceeding with build"
     exit 1
fi 


echo "Checking for changes in docs/"
echo "---"
git log -n 50 --pretty=format:"%s" -- . | grep -v '(#'
if [ $? -eq 0 ]; then
    echo "---"
    echo "✅ Changes detected in docs/ - proceeding with build"
    exit 1
else
    echo "---"
    echo "🛑 No changes detected in docs/ - ignoring build"
    exit 0
fi
