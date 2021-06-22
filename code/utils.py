import os
import sys
import subprocess
import scipy
import pandas as pd
import numpy as np
import glob
from typing import List
import datetime
import h5py
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import socket
from shutil import copyfile
from contextlib import contextmanager
import matplotlib.ticker as ticker
import jax
import jax.numpy as jnp
import torch
import collections
import json
from networkx.readwrite import json_graph


### Utils ###

# Suppress Console Output
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


# Encoder For TypeCasting Numpy Formats in DataFrame
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


# Save Graph in JSON Format
def save_graph_json(path, graph):
    data = json_graph.node_link_data(graph)
    with open(path, 'w') as f:
        json.dump(data, f, cls=MyEncoder)


# Import Graph from JSON Format
def import_graph_json(path):
    with open(path, 'r') as f:
        data = f.read()
    contents = json.loads(data)
    return json_graph.node_link_graph(contents)


# Get Extreme Nodes
def get_max_degrees(degree_sequences, node_sequences, N):
    # Lists to Fill
    max_indices = []
    max_nodes = []

    # Get Max Degrees
    sorted_deg_sequence = sorted(degree_sequences)
    max_degrees = sorted_deg_sequence[-N:]

    # GET Max Indices
    for i in range(N):
        index = degree_sequences.index(max_degrees[i])
        max_indices.append(index)

    # Get Max Nodes
    for i in range(10):
        node = node_sequences[max_indices[i]]
        max_nodes.append(node)

    # Print
    output = []
    for i in range(len(max_degrees)):
        str = "IP: {} with Degree {}".format(max_nodes[i], max_degrees[i])
        output.append(str)
        print(str + "\n")

    # Return 
    return max_degrees, max_nodes 


# Get Time Sorted CSV List
def get_csv_list(csv_dir):
    # Capture Files in Time Sorted List 
    csv_files = []
    for file in sorted(glob.glob("{}/{}".format(csv_dir, "*.csv"))):
        csv_files.append(file)

    # Return 
    return csv_files


# Remove Unwanted Protocols from DataFrame
def filter_protocols(df):
    # Dump Protocols
    dump = ['IrDA', 'USB', 'DSL', 'ISDN', 'ITU', 'ARINC', 'Ethernet', 'Bluetooth', 'ARCnet', 'ARP', 'ATM', 'CHAP', 'CDP', 'DCAP', 'DTP', 'Econet', 'FDDI', 'ITU-T', 'HDLC', 'IEEE 802.11', 'IEEE 802.16', 'LACP', 'LattisNet', 'LocalTalk', 'L2F', 'L2TP', 'LLDP', 'MAC', 'Q.710', 'NDP', 'PAgP', 'PPP', 'PPTP', 'PAP', 'RPR', 'SLIP', 'StarLAN', 'STP', 'Token Ring', 'VTP', 'VEN', 'VLAN', 'ATM', 'IS-IS', 'SPB', 'MTP', 'NSP', 'ARP', 'MPLS', 'PPPoE', 'TIPC', 'CLNP', 'IPX', 'NAT', 'Routed-SMLT', 'SCCP', 'HSRP', 'VRRP', 'IP', 'IPv4', 'IPv6', 'ICMP', 'ARP', 'RIP', 'OSPF', 'IPSEC', 'AppleTalk', 'DECnet', 'IPX', 'SPX', 'IGMP', 'IPsec']

    # Mask Entries and Remove Unwanted Protocols From DataFrame
    mask = df['Protocol'].apply(lambda x: any(item for item in dump if item in x))
    df.drop(df[mask].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Return 
    return df


# Extract Source Services by Port
def src_services(df):
     counter = 1
     for port in df['Masked Source Port'].unique():
         if int(port) == 66000:
             continue
         else:
             try:
                 print("{} {}".format(int(port), socket.getservbyport(int(port))))
                 counter += 1
             except:
                 continue


# Extract Destination Services by Port
def dst_services(df):
    counter = 1
    for port in df['Masked Destination Port'].unique():
         if int(port) == 66000:
             continue
         else:
             try:
                 print("{} {}".format(int(port), socket.getservbyport(int(port))))
                 counter += 1
             except:
                 continue 


# Extract Traffic Info by Service
def traffic_by_service(df):
    counter = 1
    i = 0
    for port in df.loc[:, 'Destination Port']:
        try:
            df.loc[i, 'Destination Port'] = socket.getservbyport(int(port))
            counter += 1
            i += 1
        except:
            i += 1
            continue
    
    return df.loc[:, 'Destination Port']


# Get Complete Data Set Time Info
def get_time_info(csv_dir):
    # Locate CSV Files
    csv_files = get_csv_list(csv_dir)

    # Extract First and Last Element
    df_start = pd.read_csv(csv_files[0])
    df_end = pd.read_csv(csv_files[-1])

    # Extract Start and End Time Info
    start_epoch = df_start.loc[0, "Time Epoch"]
    end_epoch = df_end.loc[len(df_end)-1, "Time Epoch"]

    reference = datetime.datetime(1970, 1, 1)
    start_time = reference + datetime.timedelta(0, start_epoch)
    end_time = reference + datetime.timedelta(0, end_epoch)

    # Extract Trace Duration
    duration = end_epoch - start_epoch
    d = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=int(duration))

    # Print
    print("Start of Trace: ", start_time)
    print("End of Trace: ", end_time)
    print("Total Duration of Traces: ")
    print("{} Days {} Hours {} Minutes {} Seconds".format(d.day-1, d.hour, d.minute, d.second))

    # Return
    return start_time, end_time, duration, start_epoch, end_epoch


# Get Complete Trace Time Info
def get_time_info_old(df):
    # Extract Start and End Epochs
    start_epoch = df.loc[0, "Time Epoch"]
    end_epoch = df.loc[len(df)-1, "Time Epoch"]

    # Get Start and End Date
    reference = datetime.datetime(1970, 1, 1)
    start_date = reference + datetime.timedelta(0, start_epoch)
    end_date = reference + datetime.timedelta(0, end_epoch)

    # Get Trace Duration
    duration = end_epoch - start_epoch
    d = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=int(duration))

    # Print
    print("Start of Trace: ", start_date)
    print("End of Trace: ", end_date)
    print("Total Duration of Traces: ")
    print("{} Days {} Hours {} Minutes {} Seconds".format(d.day-1, d.hour, d.minute, d.second))

    # Return
    return start_date, end_date, duration, start_epoch, end_epoch


# Get Time Epoch Info from DateTime Object
def get_epoch_info(start, end):
    # Get Start and End Epoch
    reference = datetime.datetime(1970, 1, 1)
    start_epoch = (start - reference).total_seconds()
    end_epoch = (end - reference).total_seconds()

    # Get Duration
    duration = end_epoch - start_epoch 

    # Return 
    return start_epoch, end_epoch, duration


# Get DateTime Info from Time Epoch
def get_DateTime_from_epochs(start, end):
    # Convert Epochs into DateTime Objects
    reference = datetime.datetime(1970, 1, 1)
    start_date = reference + datetime.timedelta(seconds = start)
    end_date = reference + datetime.timedelta(seconds = end)

    # Return
    return start_date, end_date


# Load all CSV Files in a single DataFrame
def load_all(csv_dir):
    # Get All CSV Files
    csv_files = get_csv_list(csv_dir)

    # Load into Pandas and Concatenate
    df = pd.concat((pd.read_csv(file) for file in csv_files), axis=0, ignore_index = True)

    # Return
    return df


# Window Slicing Function
def time_windowing(csv_path: str, time_window: List):
    # Load Data
    df = load_all(csv_path)

    # Get Time Info
    with suppress_stdout():
        start_date, end_date, trace_duration, trace_start_epoch, trace_end_epoch = get_time_info(df)

    # Check if Time Window is Valid
    window_start = time_window[0]
    window_end = time_window[-1]

    if window_start < start_date:
        window_start = start_date
        print("Adjusted Start of Time Window to: ", window_start)
    if window_end > end_date:
        window_end = end_date
        print("Adjusted End of Time Window to: ", window_end)
    else:
        None

    # Convert Window Start and Window End into Valid Time Epochs
    window_start_epoch, window_end_epoch, window_duration = get_epoch_info(window_start, window_end)

    # Find Indices in DataFrame that correspond to window_start_epoch and window_end_epoch by finding closest Time Epoch values
    all_epochs = list(df['Time Epoch'].to_numpy())
    closest_start_epoch = min(all_epochs, key=lambda x:abs(x-window_start_epoch))
    closest_end_epoch = min(all_epochs, key=lambda x:abs(x-window_end_epoch))

    loc_start = df.loc[df['Time Epoch'] == closest_start_epoch].index[0]
    loc_end = df.loc[df['Time Epoch'] == closest_end_epoch].index[0]

    locs = np.arange(loc_start, loc_end + 1)

    # Construct DataFrame of Interest with given loc_start and loc_end
    df_new = df.loc[locs]
    df_new = df_new.reset_index()
    del df_new['index']

    # Return 
    return df_new


# Window Slicing Function without Loading every CSV [Faster Performance]
def time_windowing2(csv_path: str, time_window: List):
    # Get Epochs List from CSV Path
    csv_epochs = []
    for file in sorted(glob.glob("{}/{}".format(csv_path, "*.csv"))):
        full_file_path = file
        file_only = full_file_path.split(csv_dir + '/')[1]
        epoch_only = file_only.split('-')[0]
        csv_epochs.append(int(epoch_only))

    # Get Time Info from Epochs
    start_epoch = int(csv_epochs[0])
    end_epoch = int(csv_epochs[-1])
    start_date, end_date = get_DateTime_from_epochs(start_epoch, end_epoch)

    # Convert Window Start and Window End into Valid Time Epochs
    window_start = time_window[0]
    window_end = time_window[-1]
    window_start_epoch, window_end_epoch, window_duration = get_epoch_info(window_start, window_end)

    # Check if Time Window is Valid by Converting Epochs into Integers and Comparing them with the CSV Names
    window_start_epoch = int(window_start_epoch)
    window_end_epoch = int(window_end_epoch)

    if window_start_epoch < start_epoch:
        window_start_epoch = start_epoch
        print("Adjusted Start of Time Window to: ", start_date)
    if window_end_epoch > end_epoch:
        window_end_epoch = end_epoch
        print("Adjusted End of Time Window to: ", end_date)
    else:
        None

    # Find Indices in csv_epochs that correspond to window_start_epoch and window_end_epoch by finding closest Time Epoch values
    closest_start_epoch = min(csv_epochs, key=lambda x:abs(x-window_start_epoch))
    closest_end_epoch = min(csv_epochs, key=lambda x:abs(x-window_end_epoch))

    loc_start = csv_epochs.index(closest_start_epoch)
    loc_end = csv_epochs.index(closest_end_epoch)

    # Construct DataFrame of Interest with given loc_start and loc_end
    csv_files = get_csv_list(csv_path)
    desired_csv_files = csv_files[loc_start:loc_end+1]

    df = pd.concat((pd.read_csv(file) for file in desired_csv_files), axis=0, ignore_index = True)

    # Return 
    return df


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