import os
from bing_image_downloader import downloader
list = ["Michael Jackson","Robert Downey Jr","Chris Hemsworth","Narendra Modi",'homer simpson','adelle','Abdul kalam','cristiano ronaldo','messi','donald trump']
for term in list:
  search_term = term
  folder_name = term
  downloader.download(search_term, limit=100, output_dir="persons", adult_filter_off=True, force_replace=False, timeout=60)
