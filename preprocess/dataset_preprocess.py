import os
import pandas as pd
from datasets import Dataset


def image_path(dataset, experiment, plate, well, site,  base_path):
    # 这里假设你的目录结构是：
    # base_path / dataset / experiment / f"Plate{plate}" / well / f"{well}_s{site}_w{channel}.png"
    return os.path.join(
        base_path,
        experiment,
        f"Plate{plate}-RGB",
        f"{well}_s{site}.png"
    )

BASE_PATH = r"/root/autodl-tmp/HUVEC"
META_CSV  = r"/root/autodl-tmp/rxrx1-metadata/rxrx1/metadata.csv"



def make_paths(row, suffix):
    """根据行和后缀（'_orig' or '_edit'）构造 6 个通道的路径列表"""
    paths = []
    paths.append(
        image_path(
            dataset    = row[f"dataset"],
            experiment = row[f"experiment"],
            plate      = row[f"plate"],
            site       = row[f"site"],
            well=row[f"well{suffix}"],
            base_path  = BASE_PATH
        )
    )
    return paths
def make_path(row, suffix):
    """构造单个图片的完整路径"""
    return image_path(
        dataset=row["dataset"],
        experiment=row["experiment"],
        plate=row["plate"],
        site=row["site"],
        well=row[f"well{suffix}"],
        base_path=BASE_PATH,
    )




# ———— 1. 读 Metadata ————
df = pd.read_csv(META_CSV)
df= df[ df['cell_type'] == 'HUVEC' ]
df= df[ df['dataset'] == 'test' ]
# ———— 2. 分离原始（negative）与编辑（treatment）行 ————
df_orig = df[df['well_type'] == 'negative_control']
df_edit = df[df['well_type'] == 'treatment']

keys = ['dataset','experiment','plate','site']
df_map = pd.merge(
    df_orig, df_edit,
    on=keys,
    suffixes=('_orig', '_edit')
)

print(df_map)
df_map['original_image'] = df_map.apply(lambda r: make_path(r, '_orig'), axis=1)
df_map['edited_image'  ] = df_map.apply(lambda r: make_path(r, '_edit'), axis=1)
df_map['edit_prompt'   ] = df_map['sirna_edit']   # treatment 行的 sirna


# ———— 5. 取出最终三列，构造 HuggingFace Dataset ————
df_final = df_map[['original_image', 'edit_prompt', 'edited_image']]
dataset = Dataset.from_pandas(df_final)

# ———— 6. （可选）保存到 CSV，以供后续 load_dataset("csv") ————
df_final.to_csv("validate.csv", index=False)
print("Done! 生成三列 mapping.csv：original_image, edit_prompt, edited_image")



