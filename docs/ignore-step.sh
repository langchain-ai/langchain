#!/bin/bash

echo "VERCEL_ENV: $VERCEL_ENV"
echo "VERCEL_GIT_COMMIT_REF: $VERCEL_GIT_COMMIT_REF"


if [ "$VERCEL_ENV" == "production" ] || [ "$VERCEL_GIT_COMMIT_REF" == "master" ] || [ "$VERCEL_GIT_COMMIT_REF" == "v0.1" ]; then 
    echo "✅ Production build - proceeding with build"
    exit 1; 
else 
    echo "Checking for changes in docs/ and templates/:"
    echo "---"
    git log -n 50 --pretty=format:"%s" -- . ../templates | grep -v '(#'
    if [ $? -eq 0 ]; then
        echo "---"
        echo "✅ Changes detected in docs/ or templates/ - proceeding with build"
        exit 1
    else
        echo "---"
        echo "🛑 No changes detected in docs/ or templates/ - ignoring build"
        exit 0
    fi
fi
