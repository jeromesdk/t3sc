import json
import os

import numpy as np

if __name__ == "__main__":
    curr_dir = os.getcwd()
    path_to_json = os.path.join(
        curr_dir,
        r"data\tests\2024-03-23_16-05-52\T3SC_icvl_ConstantNoise-v50_beta0_ssl0_seed0\test_metrics.json"
    )
    with open(path_to_json, 'r') as file:
        json_data = json.load(file)
    print(json_data)

    # Extracting mpsnr_in, mpsnr_out, mssim_in, mssim_out for all keys except 'n_params'
    mpsnr_in_list = [json_data[key]['mpsnr_in'] for key in json_data.keys() if key not in ['n_params', 'global']]
    mpsnr_out_list = [json_data[key]['mpsnr_out'] for key in json_data.keys() if key not in ['n_params', 'global']]
    mssim_in_list = [json_data[key]['mssim_in'] for key in json_data.keys() if key not in ['n_params', 'global']]
    mssim_out_list = [json_data[key]['mssim_out'] for key in json_data.keys() if key not in ['n_params', 'global']]

    # Calculate standard deviation
    mpsnr_in_std = np.std(mpsnr_in_list)
    mpsnr_out_std = np.std(mpsnr_out_list)
    mssim_in_std = np.std(mssim_in_list)
    mssim_out_std = np.std(mssim_out_list)

    # Print results
    print("Use the following values to have the 95% confidence interval on metrics:")
    print("2*Standard Deviation of mpsnr_in:", 2*mpsnr_in_std)
    print("2*Standard Deviation of mpsnr_out:", 2*mpsnr_out_std)
    print("2*Standard Deviation of mssim_in:", 2*mssim_in_std)
    print("2*Standard Deviation of mssim_out:", 2*mssim_out_std)
