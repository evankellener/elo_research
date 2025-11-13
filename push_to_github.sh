#!/bin/bash

# Script to automatically add, commit, and push all changes to GitHub

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔄 Updating GitHub repository...${NC}\n"

# Check if there are any changes
if [ -z "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}⚠️  No changes to commit.${NC}"
    exit 0
fi

# Show status
echo -e "${YELLOW}📋 Current status:${NC}"
git status -s
echo ""

# Get commit message
if [ -z "$1" ]; then
    # No commit message provided, prompt for one
    read -p "Enter commit message (or press Enter for default): " commit_msg
    if [ -z "$commit_msg" ]; then
        commit_msg="Update repository: $(date '+%Y-%m-%d %H:%M:%S')"
    fi
else
    # Commit message provided as argument
    commit_msg="$1"
fi

# Stage all changes
echo -e "${GREEN}📦 Staging all changes...${NC}"
git add .

# Commit
echo -e "${GREEN}💾 Committing changes...${NC}"
git commit -m "$commit_msg"

# Push to GitHub
echo -e "${GREEN}🚀 Pushing to GitHub...${NC}"
git push origin main

echo -e "\n${GREEN}✅ Successfully pushed to GitHub!${NC}"

