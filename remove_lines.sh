#!/bin/bash

# Bash script to delete all lines containing a specific string and the line below it from a text file

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <file> <string_to_remove>"
    exit 1
fi

FILE=$1
STRING=$2

# Check if the file exists
if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' does not exist."
    exit 1
fi

# Create a backup before making changes
BACKUP_FILE="${FILE}.backup"
cp "$FILE" "$BACKUP_FILE"
echo "Backup created at '$BACKUP_FILE'"

# Use sed to delete lines containing the string and the line below it
sed -i "/$STRING/{N;d}" "$FILE"

echo "Lines containing '$STRING' and the line below them have been removed from '$FILE'."
