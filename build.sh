#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install project dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --no-input

# Optional: Run migrations if you are using a database
# python manage.py migrate