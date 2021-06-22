### Import ###
import os
import sys
import subprocess
import pandas as pd
import numpy as np
import glob
from typing import List
import datetime
import h5py
from shutil import copyfile


### Utils ###

# Get Time Sorted PCAP List
def get_pcap_list(pcap_dir):
    # Capture Files in Time Sorted List 
    pcap_files = []
    for file in sorted(glob.glob("{}/{}".format(pcap_dir, "*.pcap"))):
        pcap_files.append(file)

    # Return 
    return pcap_files


# Merge IPv4 and IPv6 SRC and DST Columns 
def correct_ipv6(df):
    # Locations with NaN Values
    locs = df['ip.src'].isna()

    # Replace Values
    df.loc[locs, 'ip.src'] = df.loc[locs, 'ipv6.src']
    df.loc[locs, 'ip.dst'] = df.loc[locs, 'ipv6.dst']

    # Delete ipv6 columns
    del df['ipv6.src']
    del df['ipv6.dst']

    # Return 
    return df


# Merge UDP and TCP SRC and DST Ports
def correct_UDP_TCP_ports(df):
    # Locations with NaN Values
    locs = df['udp.srcport'].isna()

    # Replace Values
    df.loc[locs, 'udp.srcport'] = df.loc[locs, 'tcp.srcport']
    df.loc[locs, 'udp.dstport'] = df.loc[locs, 'tcp.dstport']

    # Delete TCP Columns
    del df['tcp.srcport']
    del df['tcp.dstport']

    # Return 
    return df


# Mask SRC and DST Ports larger than 1024 with value 66000
def port_masking(df):
    # Assign Masked SRC and DST Port Values in New Column
    df['Masked Source Port'] = 66000
    df['Masked Destination Port'] = 66000

    # Mask SRC Ports
    mask = df['udp.srcport'] < 1025
    df.loc[mask, 'Masked Source Port'] = df.loc[mask, 'udp.srcport']

    # Mask DST Ports
    mask = df['udp.dstport'] < 1025
    df.loc[mask, 'Masked Destination Port'] = df.loc[mask, 'udp.dstport']

    # Return
    return df


# Add Number of Packets to the DataFrame
def add_num_packets(df):
    # Retrieve Number of Packets
    num_packets = len(df)

    # Add Column 
    df['Number of Packets'] = num_packets

    # Return
    return df


# Add Trace Duration Info to the DataFrame
def add_duration_time(df):
    # Compute Duration
    duration = df.iloc[len(df)-1, df.columns.get_loc('frame.time_epoch')] - df.iloc[0, df.columns.get_loc('frame.time_epoch')]

    # Add Column
    df['Trace Duration [s]'] = duration

    # Return 
    return df
    

# Rename the DataFrame 
def rename_dataframe(df):
    # Rename Columns
    df.rename(columns = {'frame.number': 'No.', 'frame.time_epoch': 'Time Epoch', 'frame.time': 'Packet Arrival Time', 'ip.src': 'Source IP', 'ip.dst': 'Destination                           IP', '_ws.col.Protocol': 'Protocol', 'frame.len': 'Traffic Size [Byte]', 'udp.srcport': 'Source Port', 'udp.dstport': 'Destination Port'},                           inplace = True)

    # Return
    return df


# Rearrange DataFrame Columns 
def rearrange_dataframe(df):
    # New Columns
    new_cols = ['No.', 'Time Epoch', 'Packet Arrival Time', 'Trace Duration [s]', 'Number of Packets', 'Source IP', 'Destination                           IP', 'Protocol', 'Traffic Size [Byte]', 'Source Port', 'Destination Port', 'Masked Source Port', 'Masked Destination Port']

    # Rearrange
    df = df[new_cols]

    # Return
    return df


# Convert Pandas DataFrames to HDF5 
def convert_df_to_HDF(df, csv_file, csv_dir, hdf_dir):
    # File Directories 
    csv_dir = csv_dir + '/'

    # File Naming
    csv_name = csv_file.split(csv_dir)[1]
    file_name = csv_name.split('.csv')[0]

    # File Pathing
    hdf_path = hdf_dir + file_name + ('.h5')

    # Create HDF5 File
    hdf = pd.HDFStore(hdf_path)

    # Store Pandas Frame
    hdf.put('PCAP', df)

    # Close HDF5 File
    hdf.close()


# Rename CSV Files that Do Not Follow the Naming Convention of the Directory
def rename_rogue_csv_files(csv_path):
    # Get Rogue CSV Files
    rogue_csv_files = []
    _, _, filepath = next(os.walk(csv_path))

    for file in filepath:
        filename = csv_path + '/' + file
        keyword = '/capture'
        if keyword in filename:
            rogue_csv_files.append(filename)

    # Rename According to Epoch
    counter = 0
    for i in range(len(rogue_csv_files)):
        df = pd.read_csv(rogue_csv_files[i])
        epoch = int(df.loc[0, 'Time Epoch'])

        old_path = rogue_csv_files[i]
        old_name = old_path.split(csv_path + '/')[1]
        new_path = csv_path + '/' + str(epoch) + '-' + old_name

        os.rename(old_path, new_path)

        counter += 1

    print("Successfully renamed {} rogue files.".format(counter))


# Fix Broken PCAP Files
def apply_pcap_fix(pcap_file_path: str, pcap_file: str):
    # Naming
    original_name = pcap_file.split(pcap_file_path + '/')[1]
    fixed_name = 'fixed_' + original_name
    pcap_file_destination = pcap_file_path + '/' + original_name

    # Commands
    command1 = 'cd /home/djuhera/pcapfix-1.1.4'
    command2 = ('/home/djuhera/pcapfix-1.1.4/pcapfix -d {}').format(pcap_file)
    command3 = ('mv /home/djuhera/notebooks/{} {}').format(fixed_name, pcap_file_destination)

    # Execution
    subprocess.check_call(command1, shell=True)
    subprocess.run(command2, shell=True)
    subprocess.check_call(command3, shell=True)


# Remove Fraudulent PCAPs from List
def delete_fraudulent_pcaps(pcap_list, fraudulent_pcaps):
    for i in range(len(fraudulent_pcaps)):
        file_name = fraudulent_pcaps[i]
        index = pcap_list.index(file_name)
        del pcap_list[index]

    return pcap_list


# Convert PCAP Files to CSV Frames and Optionally Store as HDF5
def convert_pcap_to_csvframe(pcap_dir: str, pcap_file: str, csv_dir: str, store_HDF5: bool):
    # PCAP File Location

    # Replace File Suffix .pcap with .csv
    pcap_csv_rename = []
    for files in pcap_file:
        pcap_csv_rename.append(files.replace('pcap', 'csv'))

    # Generate List with Correct Pathing and CSV Name
    csv_file = []
    for files in pcap_csv_rename:
        csv_file.append(files.replace(pcap_dir, csv_dir))

    # TSHARK Commands
    broken_pcaps = 0
    for i in range(len(csv_file)):
        try:
            command = ('tshark -r {} -T fields '
                    '-e frame.number '
                    '-e frame.time_epoch '
                    '-e frame.time '
                    '-e ip.src '
                    '-e ipv6.src '
                    '-e ip.dst '
                    '-e ipv6.dst '
                    '-e _ws.col.Protocol '
                    '-e frame.len '
                    '-e tcp.srcport '
                    '-e tcp.dstport '
                    '-e udp.srcport '
                    '-e udp.dstport '
                    '-E header=y -E separator=, -E quote=d > {}').format(
                pcap_file[i],
                csv_file[i]
            )
            subprocess.check_call(command, shell=True)
        except: 
            print("Found Broken PCAP at index {}".format(i))
            broken_pcaps = broken_pcaps + 1
            apply_pcap_fix(pcap_dir, pcap_file[i])
            command = ('tshark -r {} -T fields '
                    '-e frame.number '
                    '-e frame.time_epoch '
                    '-e frame.time '
                    '-e ip.src '
                    '-e ipv6.src '
                    '-e ip.dst '
                    '-e ipv6.dst '
                    '-e _ws.col.Protocol '
                    '-e frame.len '
                    '-e tcp.srcport '
                    '-e tcp.dstport '
                    '-e udp.srcport '
                    '-e udp.dstport '
                    '-E header=y -E separator=, -E quote=d > {}').format(
                pcap_file[i],
                csv_file[i]
            )
            subprocess.check_call(command, shell=True)


        # Read CSV in Pandas
        pd_data = pd.read_csv(csv_file[i])

        # Apply Modifications
        correct_ipv6(pd_data)
        correct_UDP_TCP_ports(pd_data)
        port_masking(pd_data)
        add_num_packets(pd_data)
        add_duration_time(pd_data)
        rename_dataframe(pd_data)

        # Rearrange Columns
        pd_data = rearrange_dataframe(pd_data)

        # Convert Pandas Frame to CSV
        pd_data.to_csv(csv_file[i], index=False)

        # Convert Pandas Frame to HDF5 File
        if store_HDF5:
            hdf_dir = '/home/djuhera/DATA/HDF_files/'
            convert_df_to_HDF(pd_data, csv_file[i], csv_dir, hdf_dir)
        else:
            None
    
    # Rogue File Renaming
    rename_rogue_csv_files(csv_dir)

    # Final Notice
    print("Done converting {} files.".format(len(pcap_file)))
    print("Fixed {} Broken PCAPS".format(broken_pcaps))
    if store_HDF5:
        print("Files were also stored as HDF5.") 
    else:
        print("Files were not stored as HDF5.")


### Conversion ###

# PCAP Conversion + PCAP Fix
pcap_dir = "/home/djuhera/DATA/Test_Traces"
csv_dir = "/home/djuhera/DATA/CSV_files"

# Get PCAP Files
pcap_list = get_pcap_list(pcap_dir)

# Remove Fraudulent PCAPs from List
fraudulent_pcaps = ['/home/djuhera/DATA/Test_Traces/1597325710-capture-vmx6-9546.pcap', '/home/djuhera/DATA/Test_Traces/1597325747-capture-vmx6-9615.pcap', '/home/djuhera/DATA/Test_Traces/1597390653-capture-vmx6-16092.pcap', '/home/djuhera/DATA/Test_Traces/1597413418-capture-vmx6-18430.pcap']
delete_fraudulent_pcaps(pcap_list, fraudulent_pcaps)
print("Number of Files to Convert: ", len(pcap_list))

# Conversion
convert_pcap_to_csvframe(pcap_dir, pcap_list, csv_dir, store_HDF5=True)

# Finish
print("\n -------------------------------------------------- DONE --------------------------------------------------")
