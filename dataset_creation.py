import os
from PIL import Image
import pandas as pd

# Original dataset
source_path = r"D:\Milk_adulteration_images"

# New CNN dataset
target_path = r"D:\CNN_Dataset"

IMG_SIZE = (224,224)

extensions = (".bmp",".jpg",".jpeg",".png")

os.makedirs(target_path, exist_ok=True)

classes = [
    "pure_milk",
    "water",
    "sugar",
    "salt",
    "detergent"
]

counter = {}

metadata=[]

for cls in classes:

    os.makedirs(
        os.path.join(
            target_path,
            cls
        ),
        exist_ok=True
    )

    counter[cls]=1


for root,dirs,files in os.walk(source_path):

    path=root.lower()

    label=None

    if "pure_milk" in path:
        label="pure_milk"

    elif "water" in path:
        label="water"

    elif "sugar" in path:
        label="sugar"

    elif "salt" in path:
        label="salt"

    elif "detergent" in path:
        label="detergent"

    if label is None:
        continue


    concentration="unknown"

    if "1_gm" in path:
        concentration="1gm"

    elif "2_gm" in path:
        concentration="2gm"

    elif "3_gm" in path:
        concentration="3gm"

    elif "4_gm" in path:
        concentration="4gm"

    elif "5_gm" in path:
        concentration="5gm"

    elif "10%" in path:
        concentration="10pct"

    elif "20%" in path:
        concentration="20pct"

    elif "30%" in path:
        concentration="30pct"

    elif "40%" in path:
        concentration="40pct"

    elif "50%" in path:
        concentration="50pct"


    region="full"

    if "center" in path:
        region="center"

    elif "edges" in path:
        region="edges"


    for file in files:

        if file.lower().endswith(
            extensions
        ):

            old_path=os.path.join(
                root,
                file
            )

            try:

                img=Image.open(
                    old_path
                )

                img=img.resize(
                    IMG_SIZE
                )

                number=counter[
                    label
                ]

                new_name=(
                    f"{label}_"
                    f"{number:04}.jpg"
                )

                save_path=os.path.join(
                    target_path,
                    label,
                    new_name
                )

                img.convert(
                    "RGB"
                ).save(
                    save_path,
                    "JPEG",
                    quality=95
                )


                metadata.append({

                    "filename":
                    new_name,

                    "class":
                    label,

                    "concentration":
                    concentration,

                    "region":
                    region

                })


                counter[
                    label
                ]+=1


                print(
                    "Saved:",
                    new_name
                )

            except Exception as e:

                print(
                    file,
                    e
                )


df=pd.DataFrame(
    metadata
)

df.to_csv(
    os.path.join(
        target_path,
        "metadata.csv"
    ),
    index=False
)

print("\nDataset Created")
print(counter)