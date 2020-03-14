#!/usr/bin/env bash

# require kaggle.json
kaggle c download bengaliai-cv19
unzip bengaliai-cv19.zip -d ./data
rm bengaliai-cv19.zip
