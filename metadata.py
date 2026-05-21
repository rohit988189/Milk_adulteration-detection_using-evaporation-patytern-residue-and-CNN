import os
import pandas as pd

DATASET=r"D:\CNN_Dataset"

rows=[]

classes=os.listdir(DATASET)

for cls in classes:

    class_folder=os.path.join(
        DATASET,
        cls
    )

    if not os.path.isdir(class_folder):
        continue

    for file in os.listdir(class_folder):

        if file.lower().endswith(

            (".jpg",".jpeg",".png")

        ):

            relative_path=os.path.join(
                cls,
                file
            )

            rows.append({

                "filepath":relative_path,

                "class":cls,

                "concentration":"unknown",

                "region":"unknown"

            })


df=pd.DataFrame(rows)

print(df.head())

print("\nTotal:",len(df))

df.to_csv(

    "metadata.csv",

    index=False

)

print("\nmetadata.csv generated")