import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import os
import shutil
import gc

def floorify(x, lo):
    """example: x in [0, 0.01] -> x := 0"""
    return lo if x <= lo+0.01 and x >= lo else x

def floorify_zeros(x):
    """look around values [0,0.01] and determine if in proximity it's categorical. If yes - floorify"""

    x = np.asarray(x)

    has_zeros = np.any((x >= 0) & (x <= 0.01))

    no_proximity = (
        not np.any((x < 0) & (x >= -0.01)) and
        not np.any((x > 0.01) & (x <= 0.02))
    )

    if has_zeros and no_proximity:
        mask = (x >= 0) & (x <= 0.01)
        x = x.copy()
        x[mask] = 0

    return x


def floorify_ones(x):
    """look around values [1,1.01] and determine if in proximity it's categorical. If yes - floorify"""

    x = np.asarray(x)

    has_ones = np.any((x >= 1) & (x <= 1.01))

    no_proximity = (
        not np.any((x < 1) & (x >= 0.99)) and
        not np.any((x > 1.01) & (x <= 1.02))
    )

    if has_ones and no_proximity:
        mask = (x >= 1) & (x <= 1.01)
        x = x.copy()
        x[mask] = 1

    return x

def convert_to_int(x):
    """convert binary float column to Pandas Int8 if possible"""

    s = pd.Series(x)

    vals = s.dropna().unique()

    if len(vals) <= 2 and set(vals).issubset({0, 1}):
        return s.astype("Int8")

    return s


def floorify_ones_and_zeros(t):
    """apply zero-floorify, one-floorify, then convert to Int8 if binary"""

    t = floorify_zeros(t)
    t = floorify_ones(t)

    return convert_to_int(t)


def floorify_frac(x, interval=1):
    """convert to int if float appears ordinal"""
    xt = (np.floor(x/interval+1e-6))
    if np.max(xt)<=127:
        return xt.astype('Int8')
    return xt.astype('Int16')


def write_parquet_chunks(df, output_dir, chunk_size=50_000):
    
    os.makedirs(output_dir, exist_ok=True)
    prefix = output_dir.split('/')[-2].split('_')[0]
    n = len(df)
    
    for i, start in enumerate(range(0, n, chunk_size)):
        
        chunk = df.iloc[start:start+chunk_size].copy()
        
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        
        pq.write_table(
            table,
            f"{output_dir}{prefix}_part_{i:04d}.parquet",
            compression="snappy"
        )
        
        del chunk, table
        gc.collect()


# --- categorical mappings ---
d63_map = {
    'CR':0, 'XZ':1, 'XM':2, 'CO':3, 'CL':4, 'XL':5
}

d64_map = {
    '-1': -1,
    'O': 0,
    'R': 1,
    'U': 2
}

base_cat_cols = ['B_30','B_38','D_114','D_116','D_117','D_120','D_126','D_66','D_68']
cols_to_drop = ['D_87','D_88','D_108','D_110','D_111','B_39']

def convert_csv_to_parquet(csv_path, output_dir, chunk_size=50_000):

    os.makedirs(output_dir, exist_ok=True)
    prefix = output_dir.split('/')[-2].split('_')[0]
    reader = pd.read_csv(csv_path, chunksize=chunk_size, dtype={'customer_ID': "string"})
    for i, chunk in enumerate(tqdm(reader)):

        # drop sparse columns(more than 99% missing values)
        chunk = chunk.drop(cols_to_drop, axis=1, errors="ignore")
        
        # Float64 → Float32
        float_cols = chunk.select_dtypes(include=["float64"]).columns
        chunk[float_cols] = chunk[float_cols].astype("float32")
        
        
        # -- categorical transformations --
        # D_63
        if "D_63" in chunk.columns:
            chunk["D_63"] = chunk["D_63"].map(d63_map).astype("int8")

        # D_64
        if "D_64" in chunk.columns:
            chunk["D_64"] = chunk["D_64"].map(d64_map).astype("Int8")

        # categorical columns
        for col in base_cat_cols:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("Int8")

        # binary column
        if "B_31" in chunk.columns:
            chunk["B_31"] = chunk["B_31"].astype("int8")

        # can be rounded up as each bin has AUC of 0.5
        chunk['B_19'] = np.floor(chunk['B_19']*100).astype('Int8')


        chunk = chunk.copy()
        # one value overlaps, but the split can identified by S_11
        chunk.loc[chunk.S_13.between(0.67, 0.7) & (chunk.S_11.isin([15,16,17])),'S_13'] = 0.6789168283158535
        floor_vals = (0, 0.0377176456223467, 0.2804642206328049, 0.4013539714415651, 0.4206963381303189, 0.5067698438641042, 
                    0.5261121975338173, 0.5551258157960416, 0.6218568673028206, 0.6876208933830246, 0.8433269036807703, 1)
        for c in floor_vals:
            chunk['S_13'] = chunk['S_13'].apply(lambda t: floorify(t,c))
        chunk['S_13'] = np.round(chunk['S_13']*1034).astype('Int16')

        
        chunk =chunk.copy()
        # this one has many more value overlaps, but the splits can be identified by S_15
        chunk.loc[(chunk.S_8>=0.30) & (chunk.S_8<=0.35) & (chunk.S_15<=6),'S_8'] = 0.3224889650033656
        chunk.loc[(chunk.S_8>=0.30) & (chunk.S_8<=0.35) & (chunk.S_15==7),'S_8'] = 0.3145925513763017
        chunk.loc[(chunk.S_8>=0.45) & (chunk.S_8<=0.477) & (chunk.S_15==3),'S_8'] = 0.4570436553944634
        chunk.loc[(chunk.S_8>=0.45) & (chunk.S_8<=0.477) & (chunk.S_15==5),'S_8'] = 0.4636765662005172
        chunk.loc[(chunk.S_8>=0.45) & (chunk.S_8<=0.477) & (chunk.S_15==6),'S_8'] = 0.4592546209653157
        chunk.loc[(chunk.S_8>=0.55) & (chunk.S_8<=0.65) & (chunk.S_15==5),'S_8'] = 0.5938092592144236
        chunk.loc[(chunk.S_8>=0.55) & (chunk.S_8<=0.65) & (chunk.S_15==4),'S_8'] = 0.5994946974629933
        chunk.loc[(chunk.S_8>=0.55) & (chunk.S_8<=0.65) & (chunk.S_15<=2),'S_8'] = 0.6017056828901041
        chunk.loc[(chunk.S_8>=0.73) & (chunk.S_8<=0.78) & (chunk.S_15==3),'S_8'] = 0.7441567340107059
        chunk.loc[(chunk.S_8>=0.73) & (chunk.S_8<=0.78) & (chunk.S_15==5),'S_8'] = 0.7517372106519937
        chunk.loc[(chunk.S_8>=0.73) & (chunk.S_8<=0.78) & (chunk.S_15==4),'S_8'] = 0.7586861099807893
        chunk.loc[(chunk.S_8>=0.91) & (chunk.S_8<=0.98) & (chunk.S_15==4),'S_8'] = 0.9147189165383852
        chunk.loc[(chunk.S_8>=0.91) & (chunk.S_8<=0.98) & (chunk.S_15<=2),'S_8'] = 0.9327230426634736
        chunk.loc[(chunk.S_8>=0.91) & (chunk.S_8<=0.98) & (chunk.S_15==3),'S_8'] = 0.935565546481781
        chunk.loc[(chunk.S_8>=1.12) & (chunk.S_8<=1.17) & (chunk.S_15<=2),'S_8'] = 1.1440303975988897
        chunk.loc[(chunk.S_8>=1.12) & (chunk.S_8<=1.17) & (chunk.S_15==3),'S_8'] = 1.151926881019957
        floor_vals = (0, 0.1017056275625063, 0.119709415455368, 0.1667719530078215, 0.2438408100936861, 
                    0.3578648754166172, 0.4055590769093041, 0.4772583808904347, 0.4876816287061991, 
                    0.6620341135675392, 0.7005685574395781, 0.8509160456526623, 1, 1.0145299163657109, 
                    1.1051803467580654, 1.2214158871037435)
        for c in floor_vals:    
            chunk['S_8'] = chunk['S_8'].apply(lambda t: floorify(t,c))
        chunk['S_8'] = np.round(chunk['S_8']*3166).astype('Int16')


        # Split the data column into Year, Month and Day for data type conversion and size reduction.
        chunk = chunk.copy()
        date = pd.to_datetime(chunk['S_2'], errors='coerce')
        chunk = chunk.drop(['S_2'], axis=1)
        chunk[['year','month','day']] = pd.DataFrame({
            "year": date.dt.year.astype("int16"),
            "month": date.dt.month.astype("int8"),
            "day": date.dt.day.astype("int8")
        })
        del date
        
        output_path = os.path.join(output_dir, f"{prefix}_part_{i:04d}.parquet")
        chunk = chunk.copy()
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        
        pq.write_table(
            table,
            output_path,
            compression="snappy"
        )
        
        del table
        del chunk
        gc.collect()
        
# Convert test
convert_csv_to_parquet(
    "/kaggle/input/competitions/amex-default-prediction/test_data.csv",
    "/kaggle/working/test_data/"
)

# Convert train
convert_csv_to_parquet(
    "/kaggle/input/competitions/amex-default-prediction/train_data.csv",
    "/kaggle/working/train_data/"
)

X = pd.read_parquet("/kaggle/working/train_data/")

X['B_4'] = floorify_frac(X['B_4'],1/78)
X['B_16'] = floorify_frac(X['B_16'],1/12)
X['B_20'] = floorify_frac(X['B_20'],1/17)
X['B_22'] = floorify_frac(X['B_22'],1/2)
X['B_31'] = floorify_frac(X['B_31'])
X['B_32'] = floorify_frac(X['B_32'])
X['B_33'] = floorify_frac(X['B_33'])
X['B_41'] = floorify_frac(X['B_41'])


X['D_39'] = floorify_frac(X['D_39'],1/34)
X['D_44'] = floorify_frac(X['D_44'],1/8)
X['D_49'] = floorify_frac(X['D_49'],1/71)
X['D_51'] = floorify_frac(X['D_51'],1/3)
X['D_59'] = floorify_frac(X['D_59']+5/48,1/48)
X['D_65'] = floorify_frac(X['D_65'],1/38)
X['D_70'] = floorify_frac(X['D_70'],1/4)
X['D_72'] = floorify_frac(X['D_72'],1/3)
X['D_74'] = floorify_frac(X['D_74'],1/14)
X['D_75'] = floorify_frac(X['D_75'],1/15)
X['D_78'] = floorify_frac(X['D_78'],1/2)
X['D_79'] = floorify_frac(X['D_79'],1/2)
X['D_80'] = floorify_frac(X['D_80'],1/5)
X['D_81'] = floorify_frac(X['D_81'])
X['D_82'] = floorify_frac(X['D_82'],1/2)
X['D_83'] = floorify_frac(X['D_83'])
X['D_84'] = floorify_frac(X['D_84'],1/2)
X['D_86'] = floorify_frac(X['D_86'])
X['D_89'] = floorify_frac(X['D_89'],1/9)
X['D_91'] = floorify_frac(X['D_91'],1/2)
X['D_92'] = floorify_frac(X['D_92'])
X['D_93'] = floorify_frac(X['D_93'])
X['D_94'] = floorify_frac(X['D_94'])
X['D_96'] = floorify_frac(X['D_96'])
X['D_103'] = floorify_frac(X['D_103'])
X['D_106'] = floorify_frac(X['D_106'],1/23)
X['D_107'] = floorify_frac(X['D_107'],1/3)
X['D_109'] = floorify_frac(X['D_109'])
X['D_113'] = floorify_frac(X['D_113'],1/5)
X['D_122'] = floorify_frac(X['D_122'],1/7)
X['D_123'] = floorify_frac(X['D_123'])
X['D_124'] = floorify_frac(X['D_124']+1/22,1/22)
X['D_125'] = floorify_frac(X['D_125'])
X['D_126'] = floorify_frac(X['D_126']+1)
X['D_127'] = floorify_frac(X['D_127'])
X['D_129'] = floorify_frac(X['D_129'])
X['D_135'] = floorify_frac(X['D_135'])
X['D_136'] = floorify_frac(X['D_136'],1/4)
X['D_137'] = floorify_frac(X['D_137'])
X['D_138'] = floorify_frac(X['D_138'],1/2)
X['D_139'] = floorify_frac(X['D_139'])
X['D_140'] = floorify_frac(X['D_140'])
X['D_143'] = floorify_frac(X['D_143'])
X['D_145'] = floorify_frac(X['D_145'],1/11)


X['R_2'] = floorify_frac(X['R_2'])
X['R_3'] = floorify_frac(X['R_3'],1/10)
X['R_4'] = floorify_frac(X['R_4'])
X['R_5'] = floorify_frac(X['R_5'],1/2)
X['R_8'] = floorify_frac(X['R_8'])
X['R_9'] = floorify_frac(X['R_9'],1/6)
X['R_10'] = floorify_frac(X['R_10'])
X['R_11'] = floorify_frac(X['R_11'],1/2)
X['R_13'] = floorify_frac(X['R_13'],1/31)
X['R_15'] = floorify_frac(X['R_15'])
X['R_16'] = floorify_frac(X['R_16'],1/2)
X['R_17'] = floorify_frac(X['R_17'],1/35)
X['R_18'] = floorify_frac(X['R_18'],1/31)
X['R_19'] = floorify_frac(X['R_19'])
X['R_20'] = floorify_frac(X['R_20'])
X['R_21'] = floorify_frac(X['R_21'])
X['R_22'] = floorify_frac(X['R_22'])
X['R_23'] = floorify_frac(X['R_23'])
X['R_24'] = floorify_frac(X['R_24'])
X['R_25'] = floorify_frac(X['R_25'])
X['R_26'] = floorify_frac(X['R_26'],1/28)
X['R_28'] = floorify_frac(X['R_28'])


X['S_6'] = floorify_frac(X['S_6'])
X['S_11'] = floorify_frac(X['S_11']+5/25,1/25)
X['S_15'] = floorify_frac(X['S_15']+3/10,1/10)
X['S_18'] = floorify_frac(X['S_18'])
X['S_20'] = floorify_frac(X['S_20'])

cols = X.select_dtypes(include=[float]).columns
for col in tqdm(cols):
    X[col] = floorify_ones_and_zeros(X[col])
    
write_parquet_chunks(X, "/kaggle/working/train_data/")

del X

X = pd.read_parquet("/kaggle/working/test_data/")

X['B_4'] = floorify_frac(X['B_4'],1/78)
X['B_16'] = floorify_frac(X['B_16'],1/12)
X['B_20'] = floorify_frac(X['B_20'],1/17)
X['B_22'] = floorify_frac(X['B_22'],1/2)
X['B_31'] = floorify_frac(X['B_31'])
X['B_32'] = floorify_frac(X['B_32'])
X['B_33'] = floorify_frac(X['B_33'])
X['B_41'] = floorify_frac(X['B_41'])


X['D_39'] = floorify_frac(X['D_39'],1/34)
X['D_44'] = floorify_frac(X['D_44'],1/8)
X['D_49'] = floorify_frac(X['D_49'],1/71)
X['D_51'] = floorify_frac(X['D_51'],1/3)
X['D_59'] = floorify_frac(X['D_59']+5/48,1/48)
X['D_65'] = floorify_frac(X['D_65'],1/38)
X['D_70'] = floorify_frac(X['D_70'],1/4)
X['D_72'] = floorify_frac(X['D_72'],1/3)
X['D_74'] = floorify_frac(X['D_74'],1/14)
X['D_75'] = floorify_frac(X['D_75'],1/15)
X['D_78'] = floorify_frac(X['D_78'],1/2)
X['D_79'] = floorify_frac(X['D_79'],1/2)
X['D_80'] = floorify_frac(X['D_80'],1/5)
X['D_81'] = floorify_frac(X['D_81'])
X['D_82'] = floorify_frac(X['D_82'],1/2)
X['D_83'] = floorify_frac(X['D_83'])
X['D_84'] = floorify_frac(X['D_84'],1/2)
X['D_86'] = floorify_frac(X['D_86'])
X['D_89'] = floorify_frac(X['D_89'],1/9)
X['D_91'] = floorify_frac(X['D_91'],1/2)
X['D_92'] = floorify_frac(X['D_92'])
X['D_93'] = floorify_frac(X['D_93'])
X['D_94'] = floorify_frac(X['D_94'])
X['D_96'] = floorify_frac(X['D_96'])
X['D_103'] = floorify_frac(X['D_103'])
X['D_106'] = floorify_frac(X['D_106'],1/23)
X['D_107'] = floorify_frac(X['D_107'],1/3)
X['D_109'] = floorify_frac(X['D_109'])
X['D_113'] = floorify_frac(X['D_113'],1/5)
X['D_122'] = floorify_frac(X['D_122'],1/7)
X['D_123'] = floorify_frac(X['D_123'])
X['D_124'] = floorify_frac(X['D_124']+1/22,1/22)
X['D_125'] = floorify_frac(X['D_125'])
X['D_126'] = floorify_frac(X['D_126']+1)
X['D_127'] = floorify_frac(X['D_127'])
X['D_129'] = floorify_frac(X['D_129'])
X['D_135'] = floorify_frac(X['D_135'])
X['D_136'] = floorify_frac(X['D_136'],1/4)
X['D_137'] = floorify_frac(X['D_137'])
X['D_138'] = floorify_frac(X['D_138'],1/2)
X['D_139'] = floorify_frac(X['D_139'])
X['D_140'] = floorify_frac(X['D_140'])
X['D_143'] = floorify_frac(X['D_143'])
X['D_145'] = floorify_frac(X['D_145'],1/11)


X['R_2'] = floorify_frac(X['R_2'])
X['R_3'] = floorify_frac(X['R_3'],1/10)
X['R_4'] = floorify_frac(X['R_4'])
X['R_5'] = floorify_frac(X['R_5'],1/2)
X['R_8'] = floorify_frac(X['R_8'])
X['R_9'] = floorify_frac(X['R_9'],1/6)
X['R_10'] = floorify_frac(X['R_10'])
X['R_11'] = floorify_frac(X['R_11'],1/2)
X['R_13'] = floorify_frac(X['R_13'],1/31)
X['R_15'] = floorify_frac(X['R_15'])
X['R_16'] = floorify_frac(X['R_16'],1/2)
X['R_17'] = floorify_frac(X['R_17'],1/35)
X['R_18'] = floorify_frac(X['R_18'],1/31)
X['R_19'] = floorify_frac(X['R_19'])
X['R_20'] = floorify_frac(X['R_20'])
X['R_21'] = floorify_frac(X['R_21'])
X['R_22'] = floorify_frac(X['R_22'])
X['R_23'] = floorify_frac(X['R_23'])
X['R_24'] = floorify_frac(X['R_24'])
X['R_25'] = floorify_frac(X['R_25'])
X['R_26'] = floorify_frac(X['R_26'],1/28)
X['R_28'] = floorify_frac(X['R_28'])


X['S_6'] = floorify_frac(X['S_6'])
X['S_11'] = floorify_frac(X['S_11']+5/25,1/25)
X['S_15'] = floorify_frac(X['S_15']+3/10,1/10)
X['S_18'] = floorify_frac(X['S_18'])
X['S_20'] = floorify_frac(X['S_20'])

write_parquet_chunks(X, "/kaggle/working/test_data/")
del X

X = pd.read_parquet("/kaggle/working/test_data/")

float_cols = X.select_dtypes(include=[float]).columns
for col in tqdm(float_cols):
    arr = floorify_ones_and_zeros(X[col].values)
    X[col] = arr
    
write_parquet_chunks(X, "/kaggle/working/test_data/")
del X