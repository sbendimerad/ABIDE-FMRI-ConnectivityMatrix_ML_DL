#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Data download

Usage:
  download_abide.py [--pipeline=cpac] [--strategy=filt_global] [<derivative> ...]
  download_abide.py (-h | --help)

Options:
  -h --help              Show this screen
  --pipeline=cpac        Pipeline [default: cpac]
  --strategy=filt_global Strategy [default: filt_global]
  derivative             Derivatives to download

"""


import os
import urllib.request
from docopt import docopt

def download_pheno_file():
    s3_pheno_url = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
    local_pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"

    # Ensure directory exists
    os.makedirs(os.path.dirname(local_pheno_path), exist_ok=True)

    # Download phenotype file if it does not exist
    if not os.path.exists(local_pheno_path):
        print("Downloading phenotype data...")
        urllib.request.urlretrieve(s3_pheno_url, local_pheno_path)
        print("Downloaded phenotype data.")
    else:
        print("Phenotype data already exists locally.")

    return local_pheno_path

def collect_and_download(derivative, pipeline, strategy, out_dir):
    """
    Collects and downloads neuroimaging derivative files from the ABIDE dataset.
    Returns:
    None
    """
  
    s3_prefix = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
    derivative = derivative.lower()
    pipeline = pipeline.lower()
    strategy = strategy.lower()

    if "roi" in derivative:
        extension = ".1D"
    else:
        extension = ".nii.gz"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Download phenotype data if not available locally
    pheno_path = download_pheno_file()
    with open(pheno_path, "r") as s3_pheno_file:
        pheno_list = s3_pheno_file.readlines()

    header = pheno_list[0].split(",")
    file_idx = header.index("FILE_ID")

    s3_paths = []
    for pheno_row in pheno_list[1:]:
        cs_row = pheno_row.split(",")
        row_file_id = cs_row[file_idx]
        if row_file_id == "no_filename":
            continue
        filename = row_file_id + "_" + derivative + extension
        s3_path = "/".join([s3_prefix, "Outputs", pipeline, strategy, derivative, filename])
        s3_paths.append(s3_path)

    total_num_files = len(s3_paths)
    for path_idx, s3_path in enumerate(s3_paths):
        rel_path = s3_path.lstrip(s3_prefix).split("/")[-1]
        download_file = os.path.join(out_dir, rel_path)
        download_dir = os.path.dirname(download_file)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        if not os.path.exists(download_file):
            print("Retrieving: %s" % download_file)
            urllib.request.urlretrieve(s3_path, download_file)
            print("%.3f%% percent complete" % (100 * (float(path_idx + 1) / total_num_files)))
        else:
            print("File %s already exists, skipping..." % download_file)

if __name__ == "__main__":
    arguments = docopt(__doc__)

    if not arguments['<derivative>']:

        arguments['<derivative>'] = ['rois_aal', 'rois_cc200', 'rois_dosenbach160', 'rois_ez', 'rois_ho', 'rois_tt']

    pipeline = arguments.get('pipeline', 'cpac')
    strategy = arguments.get('strategy', 'filt_global')

    out_dir = os.path.abspath("data/functionals/cpac/filt_global/")

    for derivative in arguments['<derivative>']:
        collect_and_download(derivative, pipeline, strategy, os.path.join(out_dir, derivative))
