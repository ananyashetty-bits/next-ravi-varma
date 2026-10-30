#!/usr/bin/env bash
PROFILE=${1:-gen}

if [ "$PROFILE" = "gen" ]; then
    pip install -r m2kit/requirements_m2.txt
fi

pip install -e .
echo "Setup complete. Versions loaded for profile: $PROFILE"