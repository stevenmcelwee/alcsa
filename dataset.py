import pandas as pd
import numpy as np


def fill_nas(df, col_name, value):
    df = df.fillna({col_name: value})
    return df

def encode_categorical(df, cols):
    for name in cols:
        df[name] = df[name].astype('category')
        df[name] = df[name].cat.codes
    return df

def convert_longs(df, cols):
    for name in cols:
        df[name] = df[name].astype('int64')
    return df

def scale(df, cols):
    for name in cols:
        df[name] -= df[name].min()
        df[name] /= df[name].max()
    return df

def scale_all(df):
    df -= df.min()
    df /= df.max()
    return df

def analyze_df(df):
    # Do some analysis on the data
    columns = list(df.columns.values)
    columns.sort()
    print "Col,Type,Unique,Min,Max,Mean,Median"
    i = 1
    for col in columns:
        print "%s,%s,%s,%s,%s,%s,%s,%s" % (i, col, df[col].dtype, len(df[col].unique()), df[col].min(), df[col].max(), df[col].mean(), df[col].median())
        i += 1
    # must be a better way to count rows but this will work for now
    rows = df.values
    first_col = rows[:, 0]
    print "Dataset has %s rows." % len(first_col)



class NSLKDD:

    __col_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                   'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                   'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                   'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                   'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
                   'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                   'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                   'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                   'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                   'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                   'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
                   'label','cluster_id']

    __drop_cols = ['cluster_id','wrong_fragment']
    __categorical_cols = ['flag','label', 'protocol_type', 'service']

    __scale_cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                    'dst_bytes', 'land', 'urgent', 'hot',
                    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                    'num_access_files', 'is_host_login', 'is_guest_login',
                    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

    __feature_cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                   'dst_bytes', 'land', 'urgent', 'hot',
                   'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                   'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                   'num_access_files', 'is_host_login', 'is_guest_login',
                   'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                   'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                   'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                   'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                   'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                   'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

    __label_col = 'label'


    # note: cluster_id was added by researcher that posted NSL-KDD to github

    def __init__(self, data_file_path):
        self.__df = pd.read_csv(data_file_path, names=self.__col_names, encoding='latin-1')

    def preprocess(self):
        self.__df = fill_nas(self.__df, 'attack_cat', 'none')
        self.__df.drop(self.__drop_cols, 1, inplace=True)
        self.__df = encode_categorical(self.__df, self.__categorical_cols)
        self.__df = scale(self.__df, self.__scale_cols)

    def get_df(self):
        return self.__df

    def get_X(self):
        array = self.__df.values
        return array[:,0:41]

    def get_Y(self):
        array = self.__df[self.__label_col].values
        return array

    def get_feature_cols(self):
        return self.__feature_cols



class UNSW:

    __col_names = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
                   'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
                   'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
                   'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
                   'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
                   'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
                   'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm',
                   'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat',
                   'Label']

    # __drop_cols = ['attack_cat', 'Ltime','Stime']
    __drop_cols = ['Ltime','Stime']

    __categorical_cols = ['dstip','proto','service','srcip','state']

    __scale_cols = ['sport', 'dsport', 'proto', 'state', 'dur',
                    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
                    'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
                    'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
                    'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
                    'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
                    'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm',
                    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm' ]

    __feature_cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
                   'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
                   'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
                   'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
                   'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
                   'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
                   'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm',
                   'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm' ]

    __label_col = 'Label'

    def __init__(self, data_file_path):
        self.__df = pd.read_csv(data_file_path,
                         names=self.__col_names,
                         parse_dates=['Stime','Ltime'],
                         low_memory=False,
                         encoding='latin-1')

    def preprocess(self):
        self.__df.drop(self.__drop_cols, 1, inplace=True)
        self.__df = fill_nas(self.__df, 'attack_cat', 'none')
        self.__df = encode_categorical(self.__df, self.__categorical_cols)
        # hack for bad data in ports
        self.__df['dsport'] = self.__df['dsport'].apply(lambda x: np.where(x.isdigit(),x,'0')).astype('int32')
        self.__df['sport'] = self.__df['sport'].apply(lambda x: np.where(x.isdigit(),x,'0')).astype('int32')
        self.__df = scale(self.__df, self.__scale_cols)

    def get_df(self):
        return self.__df

    def get_X(self):
        array = self.__df.values
        return array[:,0:45]

    def get_Y(self):
        array = self.__df[self.__label_col].values
        return array

    def get_feature_cols(self):
        return self.__feature_cols





